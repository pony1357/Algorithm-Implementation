// SPDX-License-Identifier: GPL-2.0-only

/*
 * ============================================================================
 * conv_ftl.c — STRAW (STRess-Aware WL-based read-disturbance management)
 * 구현이 반영된 Conventional FTL for NVMeVirt SSD Emulator
 * ============================================================================
 *
 * 이 파일은 NVMeVirt [FAST'23] SSD 에뮬레이터 위에 STRAW [ASPLOS'26] 논문의
 * 핵심 메커니즘 두 가지를 구현한 것이다:
 *
 *   1) WR² (Stress-aware WL-based Read Reclaim)
 *      - 블록 단위가 아닌 워드라인(WL) 단위로 read disturbance를 모니터링
 *      - 심하게 disturb된 WL만 선별적으로 reclaim (premature RR 최소화)
 *      - Space-Saving 알고리즘을 활용한 per-WL 카운터로 공간 효율적 구현
 *
 *   2) SR² (Stress-Reduced Read)
 *      - 읽기 시 invalid WL에는 높은 Vpass, valid WL에는 낮은 Vpass 적용
 *      - per-read disturbance stress를 줄여 ERCMAX를 효과적으로 향상
 *      - update_eff_rc() 함수에서 effective read count로 모델링
 *
 * [IH Start] ~ [IH End] 주석으로 표시된 부분이 STRAW 논문 구현의 핵심이다.
 *
 * RR_MODE 매크로로 다양한 read reclaim 정책을 선택할 수 있다:
 *   - BLOCK:           블록 단위 RR (기존 Baseline)
 *   - PAGETYPE:        페이지 타입별 RR (Han et al., TVLSI'23)
 *   - COCKTAIL:        핫 페이지 재배치 기반 RR (Zhang et al., TCAD'22)
 *   - SS:              Space-Saving 카운터 기반 WL-level RR (WR² only)
 *   - ORACLE:          실제 per-WL read count 사용 (이상적인 상한 비교용)
 *   - EFFECT:          WR² + SR² (STRAW 전체)
 *   - COCKTAIL_EFFECT: STRAW + Cocktail 결합 (STRAW+Cocktail)
 *   - PAGETYPE_EFFECT: STRAW + Pagetype 결합
 * ============================================================================
 */

#include <linux/ktime.h>
#include <linux/sched/clock.h>
#include <linux/random.h>

#include "nvmev.h"
#include "conv_ftl.h"

/*
 * RR_MODE: 어떤 Read Reclaim 정책을 사용할지 결정하는 매크로.
 * EFFECT로 설정되어 있으므로 STRAW의 WR² + SR²가 모두 활성화된다.
 * (논문 §7 Table 1의 STRAW 설정에 해당)
 */
#define RR_MODE EFFECT

/* ============================================================================
 * 유틸리티 함수들
 * ============================================================================ */

/*
 * last_pg_in_wordline: 현재 PPA가 one-shot program 단위(워드라인)의
 * 마지막 페이지인지 확인한다.
 *
 * 3D TLC NAND에서 하나의 WL에는 여러 페이지(LSB/CSB/MSB)가 속하며,
 * one-shot programming은 이 페이지들을 한 번에 프로그램한다.
 * 따라서 쓰기(write) 명령은 WL의 마지막 페이지에서만 실제 NAND WRITE를 발행한다.
 */
static inline bool last_pg_in_wordline(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	return (ppa->g.pg % spp->pgs_per_oneshotpg) == (spp->pgs_per_oneshotpg - 1);
}

/*
 * should_gc / should_gc_high: GC(Garbage Collection) 트리거 조건 판단.
 * free line(블록)이 임계값 이하로 떨어지면 GC를 수행해야 한다.
 *
 * 논문 §4.1에서 언급한 바와 같이, WR²의 WL-level RR은 블록 해제를 지연시키므로
 * GC가 더 자주 발생할 수 있다 (§7.3에서 평가).
 */
static bool should_gc(struct conv_ftl *conv_ftl)
{
	return (conv_ftl->lm.free_line_cnt <= conv_ftl->cp.gc_thres_lines);
}

static inline bool should_gc_high(struct conv_ftl *conv_ftl)
{
	return conv_ftl->lm.free_line_cnt <= conv_ftl->cp.gc_thres_lines_high;
}

/* ============================================================================
 * 매핑 테이블 (Mapping Table) 관련 함수
 * ============================================================================
 * LPN(Logical Page Number) → PPA(Physical Page Address) 매핑을 관리한다.
 * maptbl: 정방향 매핑 (LPN → PPA)
 * rmap:   역방향 매핑 (PPA → LPN), GC/RR 시 valid 페이지의 LPN을 찾는데 사용
 */

/* LPN에 대응하는 PPA를 반환 */
static inline struct ppa get_maptbl_ent(struct conv_ftl *conv_ftl, uint64_t lpn)
{
	return conv_ftl->maptbl[lpn];
}

/* LPN에 대응하는 PPA를 설정 */
static inline void set_maptbl_ent(struct conv_ftl *conv_ftl, uint64_t lpn, struct ppa *ppa)
{
	NVMEV_ASSERT(lpn < conv_ftl->ssd->sp.tt_pgs);
	conv_ftl->maptbl[lpn] = *ppa;
}

/*
 * ppa2pgidx: PPA(물리 주소)를 1차원 페이지 인덱스로 변환.
 * 구조: ch → lun → plane → block → page 순서로 flat index 계산.
 * 역방향 매핑(rmap) 배열의 인덱스로 사용된다.
 */
static uint64_t ppa2pgidx(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint64_t pgidx;

	NVMEV_DEBUG_VERBOSE("%s: ch:%d, lun:%d, pl:%d, blk:%d, pg:%d\n", __func__,
			ppa->g.ch, ppa->g.lun, ppa->g.pl, ppa->g.blk, ppa->g.pg);

	pgidx = ppa->g.ch * spp->pgs_per_ch + ppa->g.lun * spp->pgs_per_lun +
		ppa->g.pl * spp->pgs_per_pl + ppa->g.blk * spp->pgs_per_blk + ppa->g.pg;

	NVMEV_ASSERT(pgidx < spp->tt_pgs);

	return pgidx;
}

/* PPA에 대응하는 LPN을 역방향 매핑에서 조회 */
static inline uint64_t get_rmap_ent(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	uint64_t pgidx = ppa2pgidx(conv_ftl, ppa);
	return conv_ftl->rmap[pgidx];
}

/* 역방향 매핑 설정: rmap[page_index(ppa)] = lpn */
static inline void set_rmap_ent(struct conv_ftl *conv_ftl, uint64_t lpn, struct ppa *ppa)
{
	uint64_t pgidx = ppa2pgidx(conv_ftl, ppa);
	conv_ftl->rmap[pgidx] = lpn;
}

/* ============================================================================
 * Victim Line 우선순위 큐 관련 콜백 함수들
 * ============================================================================
 * GC 시 어떤 line(블록)을 victim으로 선택할지 결정하기 위한 min-heap.
 * vpc(valid page count)가 낮을수록 GC 효율이 높으므로 우선순위가 높다.
 */
static inline int victim_line_cmp_pri(pqueue_pri_t next, pqueue_pri_t curr)
{
	return (next > curr);  /* vpc가 작을수록 우선순위 높음 */
}

static inline pqueue_pri_t victim_line_get_pri(void *a)
{
	return ((struct line *)a)->vpc;
}

static inline void victim_line_set_pri(void *a, pqueue_pri_t pri)
{
	((struct line *)a)->vpc = pri;
}

static inline size_t victim_line_get_pos(void *a)
{
	return ((struct line *)a)->pos;
}

static inline void victim_line_set_pos(void *a, size_t pos)
{
	((struct line *)a)->pos = pos;
}

/* ============================================================================
 * Write Flow Control (쓰기 흐름 제어)
 * ============================================================================
 * RR(Read Reclaim)로 인한 쓰기가 user write 대역폭을 과도하게 점유하지 않도록
 * credit 기반으로 제어한다. RR이 페이지를 복사(write)할 때마다 credit을 소모하고,
 * credit이 0 이하가 되면 foreground GC를 수행하여 credit을 보충한다.
 */
static inline void consume_write_credit(struct conv_ftl *conv_ftl)
{
	conv_ftl->wfc.write_credits--;
}

static void foreground_gc(struct conv_ftl *conv_ftl);

/* credit이 바닥나면 foreground GC를 반복 수행하여 보충 */
static inline void check_and_refill_write_credit(struct conv_ftl *conv_ftl)
{
	struct write_flow_control *wfc = &(conv_ftl->wfc);
	while (wfc->write_credits <= 0) {
		foreground_gc(conv_ftl);
		wfc->write_credits += wfc->credits_to_refill;
	}
}

/* ============================================================================
 * [IH Start] STRAW 구현부 — 통계 출력, 워드라인 관리, RR 정책
 * ============================================================================ */

/*
 * print_statistic: 주기적으로 (매 50,000 read I/O마다) 호출되어
 * RR로 인한 page copy 수, 총 read/write 횟수 등을 출력한다.
 * 논문 §7의 실험 평가에서 수집하는 메트릭에 대응.
 */
static void print_statistic(struct conv_ftl *conv_ftl) {
	struct wordline_mgmt *wlm = &conv_ftl->wlm;
	struct line_mgmt *lm = &conv_ftl->lm;
	NVMEV_INFO("tot rr cnt: %d\n", wlm->rr_pg_cnt);               /* 총 RR page copy 수 */
	NVMEV_INFO("tot read cnt: %lld, write cnt: %lld\n",
		lm->tt_read_cnt, lm->tt_write_cnt);                        /* 총 NAND read/write 횟수 */
	NVMEV_INFO("tot read io cnt: %lld, write io cnt: %lld\n",
		conv_ftl->tt_read_io_cnt, conv_ftl->tt_write_io_cnt);      /* 총 호스트 I/O 횟수 */
	NVMEV_INFO("tot erase cnt: %lld\n", lm->tt_erase_cnt);         /* 총 블록 erase 횟수 */
}

/*
 * print_wordline_statistic: FTL 종료 시 호출되어 채널/LUN별 워드라인 통계를 출력.
 * 각 채널/LUN에 대해:
 *   - max_rr: 가장 많이 RR된 워드라인의 RR 횟수
 *   - avg_rr: 평균 RR 횟수
 *   - max_read/avg_read: 최대/평균 read count
 *   - invalid: 4개 WL 그룹별 invalid WL 비율이 90% 이상인 횟수
 *     (논문 §5.2의 Best/Good/Bad/Worst 4그룹에 대응)
 */
static void print_wordline_statistic(struct conv_ftl *conv_ftl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct wordline_mgmt *wlm = &conv_ftl->wlm;
	struct line_mgmt *lm = &conv_ftl->lm;
	int ch, lun, blk;

	for (ch = 0; ch < spp->nchs; ch++) {
		for (lun = 0; lun < spp->luns_per_ch; lun++) {
			int max_read_cnt = 0;
			uint64_t tot_read_cnt = 0;
			int max_rr_pg_cnt = 0;
			uint64_t tot_rr_pg_cnt = 0;
			uint64_t invalid_cnt[4] = { 0, };  /* 4개 WL 그룹별 카운트 */
			for (blk = 0; blk < spp->blks_per_lun; blk++) {
				struct blk_wordline *bwl = &wlm->blk_wordlines[ch][lun][blk];
				int i;
				for (i = 0; i < spp->oneshotpgs_per_blk; i++) {
					struct wordline *wl = &wlm->blk_wordlines[ch][lun][blk].wordlines[i];
					if (wl->tot_read_cnt > max_read_cnt)
						max_read_cnt = wl->tot_read_cnt;
					tot_read_cnt += wl->tot_read_cnt;
					if (wl->rr_pg_cnt > max_rr_pg_cnt)
						max_rr_pg_cnt = wl->rr_pg_cnt;
					tot_rr_pg_cnt += wl->rr_pg_cnt;
				}
				for (i = 0; i < 4; i++)
					invalid_cnt[i] += bwl->invalid_cnt[i];
			}
			NVMEV_INFO("%d %d max_rr: %d avg_rr: %lld max_read: %d avg_read: %lld "
				"invalid: %lld %lld %lld %lld\n",
				ch, lun,
				max_rr_pg_cnt, tot_rr_pg_cnt / (spp->oneshotpgs_per_blk * spp->blks_per_lun),
				max_read_cnt, tot_read_cnt / (spp->oneshotpgs_per_blk * spp->blks_per_lun),
				invalid_cnt[0], invalid_cnt[1], invalid_cnt[2], invalid_cnt[3]);
		}
	}
	NVMEV_INFO("tot rr cnt: %d gc cnt: %d\n", wlm->rr_pg_cnt, wlm->gc_pg_cnt);
	NVMEV_INFO("tot read cnt: %lld, write cnt: %lld\n", lm->tt_read_cnt, lm->tt_write_cnt);
	NVMEV_INFO("tot read io cnt: %lld, write io cnt: %lld\n",
		conv_ftl->tt_read_io_cnt, conv_ftl->tt_write_io_cnt);
	NVMEV_INFO("tot erase cnt: %lld\n", lm->tt_erase_cnt);
}

/*
 * init_wordlines: 모든 워드라인 관리 구조체를 초기화한다.
 *
 * STRAW의 핵심 데이터 구조:
 *   - blk_wordlines[ch][lun][blk]: 블록 단위 워드라인 관리 (blk_wordline)
 *     - wordlines[]: 각 one-shot page(WL)의 상태 (read_cnt, vpc, ipc, fpc 등)
 *     - counter: Space-Saving 카운터 (논문 §6의 REC에 해당)
 *     - read_cnt: 블록-레벨 RC (논문의 RC[BLK])
 *     - eff_read_cnt: SR²에 의한 effective read count (stress 감소 반영)
 *     - iwl[4]/vwl[4]: 4개 WL 그룹별 invalid/valid 워드라인 수
 *       (논문 §5.2의 Best/Good/Bad/Worst 그룹, Figure 11)
 *     - is_rr: 현재 RR victim 리스트에 포함되어 있는지 여부
 *
 *   - rr_victim_wl_list[ch][lun]: 채널/LUN별 RR 대상 블록 리스트
 *     (블록의 RC가 GRT(Global Read Threshold)를 초과하면 이 리스트에 추가)
 *
 *   - rr_th (max_rr_th/min_rr_th): WL 그룹별 ERCMAX 임계값 범위
 *     (논문 Figure 11의 ERCMAX에 대응하며, 랜덤 변동 포함)
 *
 * WL 그룹 할당 (group_id):
 *   블록 내 WL을 4등분하여 각각 0(Best), 1(Good), 2(Bad), 3(Worst) 그룹으로 분류.
 *   이는 논문 §5.2에서 WL의 물리적 위치에 따른 reliability 차이를 반영한 것이다.
 *   - 그룹 0 (Best):  블록 앞쪽 1/4 — 가장 높은 disturbance tolerance
 *   - 그룹 3 (Worst): 블록 뒤쪽 1/4 — 가장 낮은 disturbance tolerance
 */
static void init_wordlines(struct conv_ftl *conv_ftl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	struct wordline_mgmt *wlm = &conv_ftl->wlm;
	int i = 0;
	int ch, lun, pl, blk, pg;
	struct ppa ppa = {
		.g.rsv = 0,
	};

	/* 전체 워드라인 수 = 블록당 one-shot page 수 × 전체 블록 수 */
	wlm->tt_wordlines = spp->oneshotpgs_per_blk * spp->tt_blks;
	NVMEV_ASSERT(wlm->tt_wordlines == spp->tt_pgs / spp->pgs_per_oneshotpg);

	/* 3차원 배열 할당: blk_wordlines[channel][lun][block] */
	wlm->blk_wordlines = vmalloc(sizeof(struct blk_wordline **) * spp->nchs);
	/* RR victim 리스트: 채널/LUN별로 RR 대상 블록을 linked list로 관리 */
	wlm->rr_victim_wl_list = vmalloc(sizeof(struct list_head *) * spp->nchs);

	for (ch = 0; ch < spp->nchs; ch++) {
		ppa.g.ch = ch;
		wlm->blk_wordlines[ch] = vmalloc(sizeof(struct blk_wordline *) * spp->luns_per_ch);
		wlm->rr_victim_wl_list[ch] = vmalloc(sizeof(struct list_head) * spp->luns_per_ch);

		for (lun = 0; lun < spp->luns_per_ch; lun++) {
			ppa.g.lun = lun;
			wlm->blk_wordlines[ch][lun] = vmalloc(sizeof(struct blk_wordline) * spp->blks_per_lun);

			for (pl = 0; pl < spp->pls_per_lun; pl++) {
				ppa.g.pl = pl;
				for (blk = 0; blk < spp->blks_per_pl; blk++) {
					ppa.g.blk = blk;

					/*
					 * 각 블록의 워드라인 관리 구조체 초기화.
					 *
					 * counter: Space-Saving 알고리즘 인스턴스 (논문 §6의 REC)
					 *   - NUM_SS개의 엔트리로 per-WL read count를 근사 추적
					 *   - 논문에서 32-entry REC가 기본값 (704-WL 블록 기준)
					 *   - SS 알고리즘은 read count를 과소추정하지 않아 data corruption 방지
					 *     (과대추정은 premature RR만 유발, §6.1 참조)
					 *
					 * is_rr: false로 초기화 — 아직 RR 대상이 아님
					 * read_cnt: 블록-레벨 RC, 0으로 초기화
					 * eff_read_cnt: SR²의 effective RC, 0으로 초기화
					 */
					wlm->blk_wordlines[ch][lun][blk] = (struct blk_wordline) {
						.wordlines = vmalloc(sizeof(struct wordline) * spp->oneshotpgs_per_blk),
						.entry = LIST_HEAD_INIT(wlm->blk_wordlines[ch][lun][blk].entry),
						.is_rr = false,
						.read_cnt = 0,
						.eff_read_cnt = 0,
						.invalid_cnt = { 0, },
						.counter = createSpaceSaving(NUM_SS) /* SS 카운터 생성 (NUM_SS 엔트리) */
					};

					/*
					 * iwl[4]/vwl[4] 초기화: 4개 WL 그룹별 invalid/valid WL 수.
					 *
					 * 초기 상태에서는 모든 WL이 free(= 아직 프로그램되지 않음) 이므로,
					 * iwl에 전체를 할당하고 vwl은 0으로 설정.
					 * (free WL은 valid 데이터가 없으므로 invalid로 취급)
					 *
					 * iwl_real[4]: 실제로 invalidated된 WL 수 (free가 아닌 진짜 invalid)
					 *   → REAL_INVALID 매크로로 선택적 사용
					 */
					for (i = 0; i < 4; i++) {
						wlm->blk_wordlines[ch][lun][blk].iwl[i] = spp->oneshotpgs_per_blk / 4;
						wlm->blk_wordlines[ch][lun][blk].vwl[i] = 0;
						wlm->blk_wordlines[ch][lun][blk].iwl_real[i] = 0;
					}
					/* 4로 나눠떨어지지 않는 나머지를 마지막 그룹(Worst)에 추가 */
					wlm->blk_wordlines[ch][lun][blk].iwl[3] += spp->oneshotpgs_per_blk % 4;

					/*
					 * 각 워드라인 초기화.
					 *
					 * first_ppa: 이 WL의 첫 번째 페이지의 물리 주소
					 * read_cnt:  현재 erase cycle 내에서의 read count (RC[WL])
					 * tot_read_cnt: 누적 read count (통계용, erase 후에도 유지되지 않음)
					 * rr_pg_cnt: 이 WL에서 RR로 복사된 페이지 수
					 * ipc/vpc/fpc: invalid/valid/free 페이지 수
					 *   → 초기: 모든 페이지가 free (fpc = pgs_per_oneshotpg)
					 *
					 * group_id: WL 그룹 (0=Best, 1=Good, 2=Bad, 3=Worst)
					 *   논문 §5.2, Figure 11에서 정의한 WL 그룹에 대응.
					 *   블록 내 위치에 따라 ERCMAX와 disturbance rate α가 다르다.
					 */
					for (pg = 0, i = 0; pg < spp->pgs_per_blk; pg += spp->pgs_per_oneshotpg, i++) {
						ppa.g.pg = pg;

						wlm->blk_wordlines[ch][lun][blk].wordlines[i] = (struct wordline) {
							.first_ppa = ppa,
							.read_cnt = 0,
							.tot_read_cnt = 0,
							.rr_pg_cnt = 0,
							.ipc = 0,
							.vpc = 0,
							.fpc = spp->pgs_per_oneshotpg,
						};

						/* WL 그룹 할당: 블록 내 위치를 4등분 */
						if (i < spp->oneshotpgs_per_blk / 4)
							wlm->blk_wordlines[ch][lun][blk].wordlines[i].group_id = 0; /* Best */
						else if (i < spp->oneshotpgs_per_blk / 2)
							wlm->blk_wordlines[ch][lun][blk].wordlines[i].group_id = 1; /* Good */
						else if (i < (3 * spp->oneshotpgs_per_blk) / 4)
							wlm->blk_wordlines[ch][lun][blk].wordlines[i].group_id = 2; /* Bad */
						else
							wlm->blk_wordlines[ch][lun][blk].wordlines[i].group_id = 3; /* Worst */
					}
				}
				/* 채널/LUN별 RR victim 리스트 초기화 */
				INIT_LIST_HEAD(&wlm->rr_victim_wl_list[ch][lun]);
			}
		}
	}

	wlm->rr_victim_cnt = 0;  /* 현재 RR 대상 블록 수 */
	wlm->rr_pg_cnt = 0;      /* 총 RR page copy 수 */
	wlm->gc_pg_cnt = 0;      /* 총 GC page copy 수 */

	/*
	 * ERCMAX 임계값 설정 (논문 §5.2, Figure 11 참조).
	 *
	 * EFFECT/COCKTAIL_EFFECT/PAGETYPE_EFFECT 모드에서는
	 * WL 그룹별로 다른 ERCMAX를 사용한다:
	 *   - 그룹 3 (Worst): BASE_RR_TH (가장 낮은 tolerance)
	 *   - 그룹 2 (Bad):   BASE_RR_TH × 1.3
	 *   - 그룹 1 (Good):  BASE_RR_TH × 1.6
	 *   - 그룹 0 (Best):  BASE_RR_TH × 2.0 (가장 높은 tolerance)
	 *
	 * 각 임계값에 ±ERROR_TH의 랜덤 변동을 추가하여
	 * 현실적인 chip-to-chip variation을 모사한다.
	 * (논문 Figure 10에서 관찰된 WL간 tolerance 차이 반영)
	 *
	 * 다른 모드(BLOCK, SS, ORACLE 등)에서는 단일 임계값 AVG_RR_TH를 사용.
	 */
	if (cpp->rr_mode == EFFECT || cpp->rr_mode == COCKTAIL_EFFECT || cpp->rr_mode == PAGETYPE_EFFECT) {
		wlm->max_rr_th[3] = BASE_RR_TH + ERROR_TH;                    /* Worst: 기본값 */
		wlm->min_rr_th[3] = BASE_RR_TH - ERROR_TH;
		wlm->max_rr_th[2] = ((BASE_RR_TH * 13) / 10) + ERROR_TH;     /* Bad: 1.3× */
		wlm->min_rr_th[2] = ((BASE_RR_TH * 13) / 10) - ERROR_TH;
		wlm->max_rr_th[1] = ((BASE_RR_TH * 8) / 5) + ERROR_TH;       /* Good: 1.6× */
		wlm->min_rr_th[1] = ((BASE_RR_TH * 8) / 5) - ERROR_TH;
		wlm->max_rr_th[0] = (BASE_RR_TH * 2) + ERROR_TH;             /* Best: 2.0× */
		wlm->min_rr_th[0] = (BASE_RR_TH * 2) - ERROR_TH;
	}
	else {
		/* 단일 임계값 사용 (BLOCK, SS, ORACLE 등) */
		wlm->max_rr_th[0] = AVG_RR_TH + ERROR_TH;
		wlm->min_rr_th[0] = AVG_RR_TH - ERROR_TH;
	}

	/* 랜덤 시드 초기화 — ERCMAX에 랜덤 변동을 추가하기 위함 */
	get_random_bytes(&wlm->random_base, sizeof(wlm->random_base));
	get_random_bytes(&wlm->random_add, sizeof(wlm->random_add));
}

/*
 * remove_wordlines: FTL 종료 시 워드라인 관련 메모리를 해제한다.
 * 종료 전 통계를 출력한다.
 */
static void remove_wordlines(struct conv_ftl *conv_ftl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct wordline_mgmt *wlm = &conv_ftl->wlm;
	int ch, lun, blk;

	print_wordline_statistic(conv_ftl);

	for (ch = 0; ch < spp->nchs; ch++) {
		for (lun = 0; lun < spp->luns_per_ch; lun++) {
			for (blk = 0; blk < spp->blks_per_lun; blk++) {
				vfree(wlm->blk_wordlines[ch][lun][blk].wordlines);
				freeSpaceSaving(wlm->blk_wordlines[ch][lun][blk].counter);
			}
			vfree(wlm->blk_wordlines[ch][lun]);
		}
		vfree(wlm->blk_wordlines[ch]);
	}
	vfree(wlm->blk_wordlines);
}
/* [IH End] — 워드라인 초기화/해제 */

/* ============================================================================
 * Line(슈퍼블록) 관리
 * ============================================================================
 * "line"은 모든 채널/LUN에 걸친 같은 block ID를 가진 블록들의 집합 (= 슈퍼블록).
 * 상태: free → (write) → full/victim → (GC) → free
 */

static void init_lines(struct conv_ftl *conv_ftl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *line;
	int i;

	lm->tt_lines = spp->blks_per_pl;
	NVMEV_ASSERT(lm->tt_lines == spp->tt_lines);
	lm->lines = vmalloc(sizeof(struct line) * lm->tt_lines);

	INIT_LIST_HEAD(&lm->free_line_list);
	INIT_LIST_HEAD(&lm->full_line_list);

	/* GC victim 선택을 위한 우선순위 큐 (vpc 기준 min-heap) */
	lm->victim_line_pq = pqueue_init(spp->tt_lines, victim_line_cmp_pri, victim_line_get_pri,
					 victim_line_set_pri, victim_line_get_pos,
					 victim_line_set_pos);

	lm->free_line_cnt = 0;
	for (i = 0; i < lm->tt_lines; i++) {
		lm->lines[i] = (struct line){
			.id = i,
			.ipc = 0,
			.vpc = 0,
			.pos = 0,
			.entry = LIST_HEAD_INIT(lm->lines[i].entry),
			.is_rr = false,
		};

		/* 초기에는 모든 line이 free */
		list_add_tail(&lm->lines[i].entry, &lm->free_line_list);
		lm->free_line_cnt++;
	}

	NVMEV_ASSERT(lm->free_line_cnt == lm->tt_lines);
	lm->victim_line_cnt = 0;
	lm->full_line_cnt = 0;

	/* 글로벌 통계 카운터 초기화 */
	lm->tt_read_cnt = 0;
	lm->tt_write_cnt = 0;
	lm->tt_eff_read_cnt = 0;  /* SR²로 stress가 감소된 read 횟수 */
	lm->tt_erase_cnt = 0;
}

static void remove_lines(struct conv_ftl *conv_ftl)
{
	pqueue_free(conv_ftl->lm.victim_line_pq);
	vfree(conv_ftl->lm.lines);
}

static void init_write_flow_control(struct conv_ftl *conv_ftl)
{
	struct write_flow_control *wfc = &(conv_ftl->wfc);
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	wfc->write_credits = spp->pgs_per_line;
	wfc->credits_to_refill = spp->pgs_per_line;
}

static inline void check_addr(int a, int max)
{
	NVMEV_ASSERT(a >= 0 && a < max);
}

/* free line 리스트에서 다음 빈 line을 가져온다 */
static struct line *get_next_free_line(struct conv_ftl *conv_ftl)
{
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *curline = list_first_entry_or_null(&lm->free_line_list, struct line, entry);

	if (!curline) {
		NVMEV_ERROR("No free line left in VIRT !!!!\n");
		return NULL;
	}

	list_del_init(&curline->entry);
	lm->free_line_cnt--;
	NVMEV_DEBUG("%s: free_line_cnt %d\n", __func__, lm->free_line_cnt);
	return curline;
}

/*
 * __get_wp: I/O 타입에 따라 적절한 write pointer를 반환.
 *
 * USER_IO: 호스트 쓰기용 write pointer (wp)
 * GC_IO:   GC 쓰기용 write pointer (gc_wp)
 * RR_IO:   RR 쓰기 → USER_IO와 같은 wp를 사용 (RR 데이터를 user 영역에 혼합 배치)
 *
 * 논문에서 RR_IO가 wp를 공유하는 이유:
 *   WR²의 WL-level RR은 소수의 WL만 reclaim하므로,
 *   별도의 write pointer를 유지하는 것이 비효율적.
 *   User write stream에 함께 배치하면 공간 활용도가 높아진다.
 */
static struct write_pointer *__get_wp(struct conv_ftl *ftl, uint32_t io_type)
{
	if (io_type == USER_IO) {
		return &ftl->wp;
	} else if (io_type == GC_IO) {
		return &ftl->gc_wp;
	} else if (io_type == RR_IO) {
		return &ftl->wp;  /* RR은 user write pointer를 공유 */
	}

	NVMEV_ASSERT(0);
	return NULL;
}

/* write pointer 초기 설정: 새로운 free line을 할당하고 첫 페이지를 가리킨다 */
static void prepare_write_pointer(struct conv_ftl *conv_ftl, uint32_t io_type)
{
	struct write_pointer *wp = __get_wp(conv_ftl, io_type);
	struct line *curline = get_next_free_line(conv_ftl);

	NVMEV_ASSERT(wp);
	NVMEV_ASSERT(curline);

	*wp = (struct write_pointer){
		.curline = curline,
		.ch = 0,
		.lun = 0,
		.pg = 0,
		.blk = curline->id,
		.pl = 0,
	};
}

/*
 * advance_write_pointer: write pointer를 다음 페이지로 전진시킨다.
 *
 * 전진 순서 (interleaving 최적화):
 *   page(within WL) → channel → LUN → wordline → block(new line)
 *
 * 이 순서는 채널/LUN 간 parallelism을 최대화하기 위한 것이다.
 * 하나의 WL 내 모든 페이지를 쓴 후(one-shot program 단위),
 * 다음 채널로 이동하여 인터리빙한다.
 *
 * line의 모든 페이지를 다 쓰면:
 *   - 모든 페이지가 valid이면 → full line 리스트로 이동
 *   - invalid 페이지가 있으면 → victim 큐에 삽입 (GC 대상 후보)
 *   - 새로운 free line을 할당받아 계속 쓴다
 */
static void advance_write_pointer(struct conv_ftl *conv_ftl, uint32_t io_type)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;
	struct write_pointer *wpp = __get_wp(conv_ftl, io_type);

	NVMEV_DEBUG_VERBOSE("current wpp: ch:%d, lun:%d, pl:%d, blk:%d, pg:%d\n",
			wpp->ch, wpp->lun, wpp->pl, wpp->blk, wpp->pg);

	check_addr(wpp->pg, spp->pgs_per_blk);
	wpp->pg++;

	/* WL 내 다음 페이지로 이동 (one-shot program 단위 내) */
	if ((wpp->pg % spp->pgs_per_oneshotpg) != 0)
		goto out;

	/* WL 경계 도달 → 다음 채널로 이동 */
	wpp->pg -= spp->pgs_per_oneshotpg;
	check_addr(wpp->ch, spp->nchs);
	wpp->ch++;
	if (wpp->ch != spp->nchs)
		goto out;

	/* 모든 채널 사용 완료 → 다음 LUN으로 이동 */
	wpp->ch = 0;
	check_addr(wpp->lun, spp->luns_per_ch);
	wpp->lun++;
	if (wpp->lun != spp->luns_per_ch)
		goto out;

	/* 모든 LUN 사용 완료 → 블록 내 다음 워드라인으로 이동 */
	wpp->lun = 0;
	wpp->pg += spp->pgs_per_oneshotpg;
	if (wpp->pg != spp->pgs_per_blk)
		goto out;

	/* 블록(line)의 모든 페이지를 다 사용함 → line 상태 전환 */
	wpp->pg = 0;
	if (wpp->curline->vpc == spp->pgs_per_line) {
		/* 모든 페이지가 아직 valid → full line으로 이동 */
		NVMEV_ASSERT(wpp->curline->ipc == 0);
		list_add_tail(&wpp->curline->entry, &lm->full_line_list);
		lm->full_line_cnt++;
		NVMEV_DEBUG_VERBOSE("wpp: move line to full_line_list\n");
	} else {
		/* invalid 페이지 존재 → GC victim 후보 큐에 삽입 */
		NVMEV_DEBUG_VERBOSE("wpp: line is moved to victim list\n");
		NVMEV_ASSERT(wpp->curline->vpc >= 0 && wpp->curline->vpc < spp->pgs_per_line);
		NVMEV_ASSERT(wpp->curline->ipc > 0);
		pqueue_insert(lm->victim_line_pq, wpp->curline);
		lm->victim_line_cnt++;
	}

	/* 새로운 free line 할당 */
	check_addr(wpp->blk, spp->blks_per_pl);
	wpp->curline = get_next_free_line(conv_ftl);
	NVMEV_DEBUG_VERBOSE("wpp: got new clean line %d\n", wpp->curline->id);
	wpp->blk = wpp->curline->id;
	check_addr(wpp->blk, spp->blks_per_pl);

	NVMEV_ASSERT(wpp->pg == 0);
	NVMEV_ASSERT(wpp->lun == 0);
	NVMEV_ASSERT(wpp->ch == 0);
	NVMEV_ASSERT(wpp->pl == 0);
out:
	NVMEV_DEBUG_VERBOSE("advanced wpp: ch:%d, lun:%d, pl:%d, blk:%d, pg:%d (curline %d)\n",
			wpp->ch, wpp->lun, wpp->pl, wpp->blk, wpp->pg, wpp->curline->id);
}

/* 현재 write pointer가 가리키는 위치의 PPA를 반환 (새 페이지 할당) */
static struct ppa get_new_page(struct conv_ftl *conv_ftl, uint32_t io_type)
{
	struct ppa ppa;
	struct write_pointer *wp = __get_wp(conv_ftl, io_type);

	ppa.ppa = 0;
	ppa.g.ch = wp->ch;
	ppa.g.lun = wp->lun;
	ppa.g.pg = wp->pg;
	ppa.g.blk = wp->blk;
	ppa.g.pl = wp->pl;

	NVMEV_ASSERT(ppa.g.pl == 0);

	return ppa;
}

/* ============================================================================
 * 매핑 테이블 / 역매핑 테이블 초기화 및 해제
 * ============================================================================ */

static void init_maptbl(struct conv_ftl *conv_ftl)
{
	int i;
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	conv_ftl->maptbl = vmalloc(sizeof(struct ppa) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++) {
		conv_ftl->maptbl[i].ppa = UNMAPPED_PPA;  /* 초기: 모든 LPN이 미매핑 */
	}
}

static void remove_maptbl(struct conv_ftl *conv_ftl)
{
	vfree(conv_ftl->maptbl);
}

static void init_rmap(struct conv_ftl *conv_ftl)
{
	int i;
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	conv_ftl->rmap = vmalloc(sizeof(uint64_t) * spp->tt_pgs);
	for (i = 0; i < spp->tt_pgs; i++) {
		conv_ftl->rmap[i] = INVALID_LPN;  /* 초기: 모든 PPA가 미사용 */
	}
}

static void remove_rmap(struct conv_ftl *conv_ftl)
{
	vfree(conv_ftl->rmap);
}

/* ============================================================================
 * FTL 초기화 및 해제
 * ============================================================================ */

static void conv_init_ftl(struct conv_ftl *conv_ftl, struct convparams *cpp, struct ssd *ssd)
{
	conv_ftl->cp = *cpp;        /* FTL 파라미터 복사 */
	conv_ftl->ssd = ssd;

	init_maptbl(conv_ftl);      /* LPN → PPA 매핑 테이블 */
	init_rmap(conv_ftl);        /* PPA → LPN 역매핑 테이블 */
	init_wordlines(conv_ftl);   /* [IH] STRAW 워드라인 관리 구조체 */
	init_lines(conv_ftl);       /* 슈퍼블록(line) 관리 */

	/* write pointer 초기화: user용, GC용 각각 */
	prepare_write_pointer(conv_ftl, USER_IO);
	prepare_write_pointer(conv_ftl, GC_IO);

	init_write_flow_control(conv_ftl);

	conv_ftl->tt_write_io_cnt = 0;
	conv_ftl->tt_read_io_cnt = 0;

	NVMEV_INFO("Init FTL instance with %d channels (%ld pages)\n",
		conv_ftl->ssd->sp.nchs, conv_ftl->ssd->sp.tt_pgs);
}

static void conv_remove_ftl(struct conv_ftl *conv_ftl)
{
	remove_wordlines(conv_ftl);  /* [IH] STRAW 워드라인 구조체 해제 */
	remove_lines(conv_ftl);
	remove_rmap(conv_ftl);
	remove_maptbl(conv_ftl);
}

/*
 * conv_init_params: FTL 파라미터 초기화.
 *
 * op_area_pcent: Over-Provisioning 비율
 * gc_thres_lines: GC 임계값 (free line이 이 수 이하이면 GC 트리거)
 * rr_mode: RR 정책 선택 (RR_MODE 매크로에 의해 EFFECT로 설정됨)
 */
static void conv_init_params(struct convparams *cpp)
{
	cpp->op_area_pcent = OP_AREA_PERCENT;
	cpp->gc_thres_lines = 3;
	cpp->gc_thres_lines_high = 3;
	cpp->enable_gc_delay = 1;
	cpp->pba_pcent = (int)((1 + cpp->op_area_pcent) * 100);
	cpp->rr_mode = RR_MODE;

	printk("conv_init_params rrmode: %d\n", cpp->rr_mode);
}

/* ============================================================================
 * 네임스페이스 초기화 및 해제
 * ============================================================================
 * NVMe 네임스페이스를 생성하고 SSD 파티션별 FTL 인스턴스를 초기화한다.
 * PCIe 인터페이스와 write buffer는 모든 파티션이 공유한다.
 */
void conv_init_namespace(struct nvmev_ns *ns, uint32_t id, uint64_t size, void *mapped_addr,
			 uint32_t cpu_nr_dispatcher)
{
	struct ssdparams spp;
	struct convparams cpp;
	struct conv_ftl *conv_ftls;
	struct ssd *ssd;
	uint32_t i;
	const uint32_t nr_parts = SSD_PARTITIONS;

	ssd_init_params(&spp, size, nr_parts);
	conv_init_params(&cpp);

	conv_ftls = kmalloc(sizeof(struct conv_ftl) * nr_parts, GFP_KERNEL);

	for (i = 0; i < nr_parts; i++) {
		ssd = kmalloc(sizeof(struct ssd), GFP_KERNEL);
		ssd_init(ssd, &spp, cpu_nr_dispatcher);
		conv_init_ftl(&conv_ftls[i], &cpp, ssd);
	}

	/* PCIe와 Write buffer는 첫 번째 파티션 것을 공유 */
	for (i = 1; i < nr_parts; i++) {
		kfree(conv_ftls[i].ssd->pcie->perf_model);
		kfree(conv_ftls[i].ssd->pcie);
		kfree(conv_ftls[i].ssd->write_buffer);

		conv_ftls[i].ssd->pcie = conv_ftls[0].ssd->pcie;
		conv_ftls[i].ssd->write_buffer = conv_ftls[0].ssd->write_buffer;
	}

	ns->id = id;
	ns->csi = NVME_CSI_NVM;
	ns->nr_parts = nr_parts;
	ns->ftls = (void *)conv_ftls;
	ns->size = (uint64_t)((size * 100) / cpp.pba_pcent);
	ns->mapped = mapped_addr;
	ns->proc_io_cmd = conv_proc_nvme_io_cmd;  /* I/O 핸들러 등록 */

	NVMEV_INFO("FTL physical space: %lld, logical space: %lld (physical/logical * 100 = %d)\n",
		   size, ns->size, cpp.pba_pcent);
}

void conv_remove_namespace(struct nvmev_ns *ns)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	const uint32_t nr_parts = SSD_PARTITIONS;
	uint32_t i;

	for (i = 1; i < nr_parts; i++) {
		conv_ftls[i].ssd->pcie = NULL;
		conv_ftls[i].ssd->write_buffer = NULL;
	}

	for (i = 0; i < nr_parts; i++) {
		conv_remove_ftl(&conv_ftls[i]);
		ssd_remove(conv_ftls[i].ssd);
		kfree(conv_ftls[i].ssd);
	}

	kfree(conv_ftls);
	ns->ftls = NULL;
}

/* ============================================================================
 * PPA 유효성 검증 유틸리티
 * ============================================================================ */

static inline bool valid_ppa(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	int ch = ppa->g.ch;
	int lun = ppa->g.lun;
	int pl = ppa->g.pl;
	int blk = ppa->g.blk;
	int pg = ppa->g.pg;

	if (ch < 0 || ch >= spp->nchs) return false;
	if (lun < 0 || lun >= spp->luns_per_ch) return false;
	if (pl < 0 || pl >= spp->pls_per_lun) return false;
	if (blk < 0 || blk >= spp->blks_per_pl) return false;
	if (pg < 0 || pg >= spp->pgs_per_blk) return false;

	return true;
}

static inline bool valid_lpn(struct conv_ftl *conv_ftl, uint64_t lpn)
{
	return (lpn < conv_ftl->ssd->sp.tt_pgs);
}

static inline bool mapped_ppa(struct ppa *ppa)
{
	return !(ppa->ppa == UNMAPPED_PPA);
}

static inline struct line *get_line(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	return &(conv_ftl->lm.lines[ppa->g.blk]);
}

/* ============================================================================
 * [IH Start] STRAW 핵심 함수들 — WL 조회, SR², RR 판단
 * ============================================================================ */

/*
 * get_blk_wordline: PPA에 해당하는 블록의 blk_wordline 구조체를 반환.
 * blk_wordline은 블록 단위의 워드라인 관리 정보를 담고 있다:
 *   - 블록-레벨 read_cnt (RC[BLK])
 *   - Space-Saving 카운터 (REC)
 *   - WL 그룹별 invalid/valid WL 수
 *   - SR²의 effective read count
 */
static inline struct blk_wordline *get_blk_wordline(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	return &(conv_ftl->wlm.blk_wordlines[ppa->g.ch][ppa->g.lun][ppa->g.blk]);
}

/*
 * get_wordline: PPA에 해당하는 개별 워드라인(wordline) 구조체를 반환.
 * pg를 pgs_per_oneshotpg로 나누어 one-shot page(WL) 인덱스를 계산한다.
 */
static inline struct wordline *get_wordline(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	return &(conv_ftl->wlm.blk_wordlines[ppa->g.ch][ppa->g.lun][ppa->g.blk]
		.wordlines[ppa->g.pg / conv_ftl->ssd->sp.pgs_per_oneshotpg]);
}

/*
 * get_invalid_wl_ratio: 특정 WL 그룹 내에서 invalid WL의 비율(%)을 반환.
 *
 * 이 값은 SR²의 Vpass-reduction model (논문 Figure 13)에서
 * RINV(invalid non-adjacent WL 비율)에 대응한다.
 *
 * REAL_INVALID 매크로에 따라:
 *   - 1이면: 실제로 invalidate된 WL만 카운트 (free WL 제외)
 *   - 0이면: free WL도 invalid로 카운트 (보수적 추정)
 */
static int get_invalid_wl_ratio(struct conv_ftl *conv_ftl, struct ppa *ppa, int group_num)
{
	struct blk_wordline *bwl = get_blk_wordline(conv_ftl, ppa);
#if REAL_INVALID == 1
	if (bwl->iwl_real[group_num] + bwl->vwl[group_num] > 0)
		return (bwl->iwl_real[group_num] * 100) / (bwl->iwl_real[group_num] + bwl->vwl[group_num]);
	else
		return 0;
#else
	return (bwl->iwl[group_num] * 100) / (bwl->iwl[group_num] + bwl->vwl[group_num]);
#endif
}

/*
 * is_wl_invalid: 워드라인의 모든 페이지가 invalid(또는 free)인지 확인.
 * vpc(valid page count)가 0이면 해당 WL에는 유효한 데이터가 없다.
 *
 * SR²에서 invalid WL에는 높은 Vpass를 적용할 수 있다 (논문 §4.2, Figure 9).
 */
static bool is_wl_invalid(struct wordline *wl)
{
	return wl->vpc == 0;
}

/*
 * update_eff_rc: SR² (Stress-Reduced Read)의 effective read count를 갱신한다.
 *
 * ===== 논문 대응: §4.2 Stress-Reduced Read (SR²) =====
 *
 * SR²의 핵심 아이디어:
 *   - invalid WL에 더 높은 Vpass를 적용하면 target WL의 RBER이 감소함 (NERR↓)
 *   - 이 여유분(margin)을 활용하여 valid WL에 더 낮은 Vpass를 적용
 *   - 낮은 Vpass는 read disturbance stress를 지수적으로 감소시킴
 *     (논문 Figure 14: ΔVpass = -5%일 때 per-read stress 38% 감소)
 *
 * 구현 방식:
 *   기본 effective_rc_per_read = 10 (default Vpass에서의 per-read stress 단위)
 *
 *   1) 인접 WL 기반 stress 감소 (Adjacent WL validity):
 *      - 대상 WL의 위/아래 인접 WL이 모두 invalid이면 stress 3 감소
 *      - 논문 §5.3에서 인접 WL에 높은 Vpass 적용 시 NERR 33% 감소에 대응
 *      - 인접 WL은 VpassH(10% 높은 전압)를 적용받으므로 영향이 큼
 *
 *   2) 비인접 WL 기반 stress 감소 (Non-adjacent WL invalid ratio, RINV):
 *      - 블록 내 4개 WL 그룹 각각에서 invalid WL 비율 ≥ 90%인 그룹 수에 따라
 *        추가로 stress를 2~4 감소
 *      - 논문 Figure 13의 Vpass-reduction model에 대응
 *      - RINV가 높을수록 더 많은 invalid WL에 높은 Vpass를 적용할 수 있고,
 *        그만큼 valid WL의 Vpass를 더 크게 낮출 수 있음
 *
 *   최종 eff_read_cnt += effective_rc_per_read
 *     → 이 값이 작을수록 SR²가 더 많은 stress를 줄인 것
 *     → should_reclaim_wl_ss_effective()에서 eff_read_cnt/10을 blk_read_cnt로 사용
 *     → 결과적으로 ERCMAX에 도달하기까지 더 많은 읽기가 가능해짐
 */
static void update_eff_rc(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;

	/* 기본 stress: 10 단위 (default Vpass에서의 1회 read disturbance) */
	unsigned int effective_rc_per_read = 10;

	struct blk_wordline *bwl = get_blk_wordline(conv_ftl, ppa);
	int oneshotpg = ppa->g.pg / spp->pgs_per_oneshotpg;

	/*
	 * Step 1: 인접 WL의 validity에 따른 stress 감소.
	 *
	 * 대상 WL(target)의 위/아래 인접 WL이 모두 invalid이면,
	 * SR²가 인접 WL에 VpassH(+10%)를 적용하여 NERR을 줄일 수 있다.
	 * 이 margin을 활용해 target WL의 Vpass를 낮추면 stress가 감소한다.
	 *
	 * 첫 번째 WL(oneshotpg == 0)과 마지막 WL은 한쪽 인접 WL만 존재하므로
	 * 별도 처리. (현재 구현에서는 첫/마지막 WL에 대해 감소를 적용하지 않음)
	 */
	if (oneshotpg == 0) {
		/* 블록의 첫 번째 WL — 아래쪽 인접 WL만 존재 (현재 미적용) */
	}
	else if (oneshotpg == spp->oneshotpgs_per_blk - 1) {
		/* 블록의 마지막 WL — 위쪽 인접 WL만 존재 (현재 미적용) */
	}
	else {
		/* 중간 WL: 위/아래 인접 WL 모두 invalid이면 stress 감소 */
		if (is_wl_invalid(&(bwl->wordlines[oneshotpg - 1])) &&
		    is_wl_invalid(&(bwl->wordlines[oneshotpg + 1])))
		{
			effective_rc_per_read -= 3;  /* 인접 WL invalid → -3 stress */
			lm->tt_eff_read_cnt++;       /* 통계: SR²로 감소된 read 횟수 */
		}
	}

	/*
	 * Step 2: 비인접 WL의 invalid 비율(RINV)에 따른 stress 감소.
	 *
	 * 4개 WL 그룹(Best/Good/Bad/Worst) 각각에서 invalid WL 비율을 확인.
	 * 비율이 90% 이상인 그룹이 많을수록 더 많은 invalid WL에
	 * 높은 Vpass를 적용할 수 있으므로 stress를 더 크게 줄인다.
	 *
	 * 논문 Figure 13의 Vpass-reduction model:
	 *   RINV < 0.25: ΔVpass 감소 불가
	 *   RINV < 0.50: -2.5% 가능 (일부 조건)
	 *   RINV < 0.75: -2.5%~5% 가능
	 *   RINV < 1.00: -2.5%~5% 가능 (최대 효과)
	 *
	 * invalid_cnt에 따른 stress 감소:
	 *   0개 그룹: 감소 없음 (RINV 너무 낮음)
	 *   1개 그룹: -2 (RINV ~25%)
	 *   2개 그룹: -2 (RINV ~50%)
	 *   3개 그룹: -3 (RINV ~75%)
	 *   4개 그룹: -4 (RINV ~100%, 최대 Vpass 감소 가능)
	 */
	int invalid_cnt = 0;
	int group_num;
	for (group_num = 0; group_num < 4; group_num++) {
		int invalid_WL_ratio = get_invalid_wl_ratio(conv_ftl, ppa, group_num);
		if (invalid_WL_ratio >= 90)
			invalid_cnt++;
	}
	switch (invalid_cnt) {
	case 0:
		break;
	case 1:
		bwl->invalid_cnt[0]++;   /* 통계: 1개 그룹 90%+ invalid 횟수 */
		effective_rc_per_read -= 2;
		break;
	case 2:
		bwl->invalid_cnt[1]++;
		effective_rc_per_read -= 2;
		break;
	case 3:
		bwl->invalid_cnt[2]++;
		effective_rc_per_read -= 3;
		break;
	case 4:
		bwl->invalid_cnt[3]++;
		effective_rc_per_read -= 4;
		break;
	}

	/* 최종 effective read count 누적 (블록 단위) */
	bwl->eff_read_cnt += effective_rc_per_read;
}

/*
 * get_random_ppa: SSD 전체 범위에서 랜덤 PPA를 생성한다.
 *
 * Cocktail 모드 (논문 §7, Zhang et al. TCAD'22)에서 사용:
 *   Cocktail은 RR 시 핫 페이지를 랜덤 블록으로 재배치하여
 *   블록 간 read count를 균형 맞추는 전략이다.
 *   check_rr()에서 현재 읽는 PPA 대신 랜덤 PPA의 블록을 체크함으로써
 *   RR 트리거를 분산시킨다.
 */
struct ppa get_random_ppa(struct conv_ftl *conv_ftl) {
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	int random_addr;
	struct ppa res_ppa;
	get_random_bytes(&random_addr, sizeof(random_addr));
	random_addr %= spp->tt_pgs;
	res_ppa.g.ch = random_addr / spp->pgs_per_ch;
	random_addr %= spp->pgs_per_ch;
	res_ppa.g.lun = random_addr / spp->pgs_per_lun;
	random_addr %= spp->pgs_per_lun;
	res_ppa.g.pl = random_addr / spp->pgs_per_pl;
	random_addr %= spp->pgs_per_pl;
	res_ppa.g.blk = random_addr / spp->pgs_per_blk;
	random_addr %= spp->pgs_per_blk;
	res_ppa.g.pg = random_addr;

	return res_ppa;
}

/*
 * check_rr: 매 읽기 시 호출되어 read disturbance를 추적하고 RR 트리거 여부를 판단.
 *
 * ===== 논문 대응: §6 StrawFTL Operational Overview (Figure 15) =====
 *
 * 동작 흐름:
 *   1) RR 모드에 따라 target PPA 결정
 *      - COCKTAIL/COCKTAIL_EFFECT: 랜덤 PPA 사용 (핫 페이지 분산)
 *      - 나머지: 실제 읽은 PPA 사용
 *
 *   2) RR 모드에 따라 카운터 업데이트
 *      - BLOCK/PAGETYPE/COCKTAIL/ORACLE: 블록/WL read count만 증가
 *      - SS: Space-Saving 카운터에 WL index 추가 (per-WL RC 근사 추적)
 *      - EFFECT/PAGETYPE_EFFECT/COCKTAIL_EFFECT:
 *        a) update_eff_rc()로 SR²의 effective RC 갱신
 *        b) Space-Saving 카운터에 WL index 추가
 *
 *   3) 블록-레벨 read count (bwl->read_cnt) 증가
 *      WL-레벨 read count (wl->read_cnt, wl->tot_read_cnt) 증가
 *
 *   4) 블록 RC가 GRT(Global Read Threshold) 배수에 도달하면:
 *      → 해당 블록을 RR victim 리스트에 추가
 *      (논문 Figure 15의 "RC[BLKtgt] % itv. == 0?" 조건에 대응)
 *
 * GRT는 논문에서 "predefined interval (e.g., 1K reads)"에 해당하며,
 * 이 주기마다 블록 내 모든 WL을 검사하여 heavily disturbed WL을 식별한다.
 */
static void check_rr(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct convparams *cpp = &conv_ftl->cp;
	struct wordline *wl;
	struct blk_wordline *bwl;
	struct ppa random_ppa;
	struct ppa *target_ppa;

	/* Cocktail 모드: 랜덤 블록에 대해 RR 체크 (핫 페이지 분산 효과) */
	if (cpp->rr_mode == COCKTAIL || cpp->rr_mode == COCKTAIL_EFFECT) {
		random_ppa = get_random_ppa(conv_ftl);
		target_ppa = &random_ppa;
	}
	else {
		target_ppa = ppa;
	}

	wl = get_wordline(conv_ftl, target_ppa);
	bwl = get_blk_wordline(conv_ftl, target_ppa);

	/*
	 * RR 모드별 카운터 업데이트:
	 *
	 * BLOCK/PAGETYPE/COCKTAIL/ORACLE:
	 *   블록/WL read count만 증가 (SS 카운터 미사용)
	 *
	 * SS:
	 *   Space-Saving 카운터에 현재 WL index를 add (per-WL RC 근사 추적)
	 *   → 논문 §6의 REC (Resource-Efficient Counter)
	 *
	 * EFFECT/PAGETYPE_EFFECT/COCKTAIL_EFFECT:
	 *   a) update_eff_rc(): SR²의 effective RC 갱신 (Vpass 감소 효과 반영)
	 *   b) SS 카운터에 WL index 추가
	 */
	switch (cpp->rr_mode) {
	case BLOCK:
	case PAGETYPE:
	case COCKTAIL:
	case ORACLE:
		break;  /* SS 카운터 미사용 */
	case SS:
		/* Space-Saving 카운터에 현재 WL index 추가 */
		add(bwl->counter, target_ppa->g.pg / conv_ftl->ssd->sp.pgs_per_oneshotpg);
		break;
	case EFFECT:
	case PAGETYPE_EFFECT:
	case COCKTAIL_EFFECT:
		/* SR²: effective read count 갱신 + SS 카운터 업데이트 */
		update_eff_rc(conv_ftl, target_ppa);
		add(bwl->counter, target_ppa->g.pg / conv_ftl->ssd->sp.pgs_per_oneshotpg);
		break;
	}

	/* 블록-레벨 및 WL-레벨 read count 증가 */
	bwl->read_cnt++;
	wl->read_cnt++;
	wl->tot_read_cnt++;

	/*
	 * GRT(Global Read Threshold) 도달 시 RR victim 리스트에 추가.
	 *
	 * 블록의 read_cnt가 GRT의 배수에 도달하면, 해당 블록에 대해
	 * WL-level 검사를 수행할 필요가 있음을 표시한다.
	 *
	 * 논문 Figure 15: "RC[BLKtgt] % itv. == 0?" → Yes → WL 검사
	 *
	 * is_rr == false인 블록만 victim 리스트에 추가 (중복 방지).
	 * flash_seq = 0으로 초기화: RR 진행 시 블록 내 flashpage를 순차적으로 처리.
	 */
	if (bwl->read_cnt % GRT == 0) {
		if (bwl->is_rr == false) {
			struct wordline_mgmt *wlm = &conv_ftl->wlm;
			list_del_init(&bwl->entry);
			bwl->is_rr = true;
			bwl->flash_seq = 0;  /* RR 진행 인덱스 초기화 */
			list_add_tail(&bwl->entry,
				&wlm->rr_victim_wl_list[target_ppa->g.ch][target_ppa->g.lun]);
			wlm->rr_victim_cnt++;
		}
	}
}
/* [IH End] — check_rr */

/* ============================================================================
 * 페이지 상태 관리: valid ↔ invalid ↔ free
 * ============================================================================ */

/*
 * mark_page_invalid: valid 페이지를 invalid로 전환한다.
 * 호출 시점: overwrite (동일 LPN에 새로운 데이터 쓰기) 또는 RR 시 old page 무효화.
 *
 * 업데이트 대상:
 *   1) NAND page 상태: PG_VALID → PG_INVALID
 *   2) 워드라인 상태: wl->vpc--, wl->ipc++
 *      → WL의 모든 페이지가 invalid이 되면 (vpc == 0):
 *        해당 WL이 속한 그룹의 iwl/vwl 카운터 업데이트
 *        (SR²에서 invalid WL에 높은 Vpass를 적용할 수 있게 됨)
 *   3) NAND block 상태: blk->vpc--, blk->ipc++
 *   4) Line(슈퍼블록) 상태: line->vpc--, line->ipc++
 *      → full line이었으면 victim 큐로 이동 (GC 대상 후보)
 */
static void mark_page_invalid(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;
	struct nand_block *blk = NULL;
	struct nand_page *pg = NULL;
	struct wordline *wl = NULL;
	bool was_full_line = false;
	struct line *line;

	/* 1) 페이지 상태 전환 */
	pg = get_pg(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(pg->status == PG_VALID);
	pg->status = PG_INVALID;

	/* 2) 워드라인 상태 업데이트 */
	wl = get_wordline(conv_ftl, ppa);
	NVMEV_ASSERT((wl->ipc + wl->vpc + wl->fpc) == spp->pgs_per_oneshotpg);
	wl->ipc++;
	wl->vpc--;
	if (wl->vpc == 0) {
		/*
		 * WL의 모든 valid 페이지가 invalid이 됨 → WL 전체가 invalid.
		 * 이 WL 그룹의 iwl(invalid WL 수) 증가, vwl(valid WL 수) 감소.
		 * SR²에서 이 WL에 높은 Vpass를 적용할 수 있게 된다.
		 */
		struct blk_wordline *bwl = get_blk_wordline(conv_ftl, ppa);
		bwl->iwl[wl->group_id]++;
		bwl->iwl_real[wl->group_id]++;
		bwl->vwl[wl->group_id]--;
	}

	/* 3) 블록 상태 업데이트 */
	blk = get_blk(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(blk->ipc >= 0 && blk->ipc < spp->pgs_per_blk);
	blk->ipc++;
	NVMEV_ASSERT(blk->vpc > 0 && blk->vpc <= spp->pgs_per_blk);
	blk->vpc--;

	/* 4) Line 상태 업데이트 */
	line = get_line(conv_ftl, ppa);
	NVMEV_ASSERT(line->ipc >= 0 && line->ipc < spp->pgs_per_line);
	if (line->vpc == spp->pgs_per_line) {
		NVMEV_ASSERT(line->ipc == 0);
		was_full_line = true;
	}
	line->ipc++;
	NVMEV_ASSERT(line->vpc > 0 && line->vpc <= spp->pgs_per_line);

	/* victim 큐에 이미 있는 line이면 우선순위(vpc) 업데이트 */
	if (line->pos) {
		pqueue_change_priority(lm->victim_line_pq, line->vpc - 1, line);
	} else {
		line->vpc--;
	}

	/* full line이었으면 → victim 큐로 이동 */
	if (was_full_line) {
		list_del_init(&line->entry);
		lm->full_line_cnt--;
		pqueue_insert(lm->victim_line_pq, line);
		lm->victim_line_cnt++;
	}
}

/*
 * mark_page_valid: free 페이지에 데이터를 프로그램하여 valid로 전환.
 * 호출 시점: user write, GC write, RR write 시 새 페이지에 데이터를 쓸 때.
 *
 * WL의 첫 번째 valid 페이지가 프로그램되면 (vpc: 0→1),
 * 해당 WL 그룹의 iwl 감소, vwl 증가.
 * (이전에 invalid(또는 free)였던 WL이 valid 데이터를 갖게 됨)
 */
static void mark_page_valid(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct nand_block *blk = NULL;
	struct nand_page *pg = NULL;
	struct wordline *wl = NULL;
	struct line *line;

	pg = get_pg(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(pg->status == PG_FREE);
	pg->status = PG_VALID;

	wl = get_wordline(conv_ftl, ppa);
	NVMEV_ASSERT((wl->ipc + wl->vpc + wl->fpc) == spp->pgs_per_oneshotpg);
	wl->fpc--;
	wl->vpc++;
	if (wl->vpc == 1) {
		/* WL이 처음으로 valid 데이터를 가지게 됨 → 그룹 카운터 업데이트 */
		struct blk_wordline *bwl = get_blk_wordline(conv_ftl, ppa);
		bwl->iwl[wl->group_id]--;
		bwl->vwl[wl->group_id]++;
	}

	blk = get_blk(conv_ftl->ssd, ppa);
	NVMEV_ASSERT(blk->vpc >= 0 && blk->vpc < spp->pgs_per_blk);
	blk->vpc++;

	line = get_line(conv_ftl, ppa);
	NVMEV_ASSERT(line->vpc >= 0 && line->vpc < spp->pgs_per_line);
	line->vpc++;
}

/*
 * mark_block_free: 블록 erase 후 모든 상태를 초기화한다.
 *
 * 모든 페이지 → PG_FREE, 블록/WL 카운터 리셋, SS 카운터 리셋.
 * 논문 Figure 15: "Is BLKtgt empty? → Yes → Erase BLKtgt, Reset counters"
 *
 * WL 그룹별 iwl/vwl도 초기 상태로 복원:
 *   모든 WL이 free이므로 iwl = oneshotpgs/4, vwl = 0.
 */
static void mark_block_free(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct nand_block *blk = get_blk(conv_ftl->ssd, ppa);
	struct nand_page *pg = NULL;
	struct blk_wordline *bwl = get_blk_wordline(conv_ftl, ppa);
	int i;

	/* 모든 페이지 상태 초기화 */
	for (i = 0; i < spp->pgs_per_blk; i++) {
		pg = &blk->pg[i];
		NVMEV_ASSERT(pg->nsecs == spp->secs_per_pg);
		pg->status = PG_FREE;
	}

	/* 블록 카운터 초기화 */
	NVMEV_ASSERT(blk->npgs == spp->pgs_per_blk);
	blk->ipc = 0;
	blk->vpc = 0;
	blk->erase_cnt++;

	/* WL 그룹별 invalid/valid 카운터 초기화 (free = invalid 취급) */
	for (i = 0; i < 4; i++) {
		bwl->iwl[i] = spp->oneshotpgs_per_blk / 4;
		bwl->vwl[i] = 0;
		bwl->iwl_real[i] = 0;
		if (i == 3)
			bwl->iwl[i] += spp->oneshotpgs_per_blk % 4;
	}

	/* 블록-레벨 카운터 리셋 (다음 erase cycle 시작) */
	bwl->read_cnt = 0;
	bwl->eff_read_cnt = 0;

	/* 각 워드라인 카운터 리셋 */
	for (i = 0; i < spp->oneshotpgs_per_blk; i++) {
		struct wordline *wl = &(bwl->wordlines[i]);
		NVMEV_ASSERT((wl->ipc + wl->vpc + wl->fpc) == spp->pgs_per_oneshotpg);
		wl->fpc = spp->pgs_per_oneshotpg;
		wl->vpc = 0;
		wl->ipc = 0;
		wl->read_cnt = 0;
	}

	/* Space-Saving 카운터 리셋 (다음 erase cycle을 위해) */
	reset(bwl->counter);
}

/* ============================================================================
 * GC (Garbage Collection) 관련 함수
 * ============================================================================ */

/* GC 읽기: victim 블록에서 valid 페이지를 읽는다 (NAND latency 시뮬레이션) */
static void gc_read_page(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	if (cpp->enable_gc_delay) {
		struct nand_cmd gcr = {
			.type = GC_IO,
			.cmd = NAND_READ,
			.stime = 0,
			.xfer_size = spp->pgsz,
			.interleave_pci_dma = false,
			.ppa = ppa,
		};
		ssd_advance_nand(conv_ftl->ssd, &gcr);
	}
}

/*
 * gc_write_page: valid 페이지를 old 위치에서 new 위치로 복사한다.
 * GC와 RR 모두에서 사용된다 (type 파라미터로 구분).
 *
 * [IH] type == RR_IO일 때의 차이점:
 *   - old page를 즉시 invalid로 마킹 (WR²의 WL-level reclaim)
 *     → 블록-level RR과 달리, 블록 전체가 아닌 특정 WL의 페이지만 이동
 *   - rr_pg_cnt 증가 (RR-induced page copy 통계)
 *   - write credit 소모 (RR도 쓰기 대역폭을 사용하므로)
 *
 * type == GC_IO일 때:
 *   - old page는 블록 erase 시 일괄 처리되므로 여기서 invalid 마킹 불필요
 *   - gc_pg_cnt 증가
 */
static uint64_t gc_write_page(struct conv_ftl *conv_ftl, struct ppa *old_ppa, int type)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	struct ppa new_ppa;
	uint64_t lpn = get_rmap_ent(conv_ftl, old_ppa);

	NVMEV_ASSERT(valid_lpn(conv_ftl, lpn));
	new_ppa = get_new_page(conv_ftl, type);

	/* 매핑 테이블 업데이트: LPN → new PPA */
	set_maptbl_ent(conv_ftl, lpn, &new_ppa);
	/* 역매핑 업데이트: new PPA → LPN */
	set_rmap_ent(conv_ftl, lpn, &new_ppa);

	mark_page_valid(conv_ftl, &new_ppa);

	if (type == RR_IO) {
		/*
		 * RR (Read Reclaim): WL-level에서 선별적으로 페이지 이동.
		 * old page를 즉시 invalid로 마킹하고 역매핑을 INVALID_LPN으로 설정.
		 * 이는 WR²의 핵심 — 블록 전체를 erase하지 않고 WL 단위로 reclaim.
		 */
		mark_page_invalid(conv_ftl, old_ppa);
		set_rmap_ent(conv_ftl, INVALID_LPN, old_ppa);
		conv_ftl->wlm.rr_pg_cnt++;  /* 논문 Figure 17의 "RR-induced page copies" */
	}
	else if (type == GC_IO) {
		conv_ftl->wlm.gc_pg_cnt++;
	}

	/* write pointer 전진 */
	advance_write_pointer(conv_ftl, type);

	/* NAND write latency 시뮬레이션 (WL의 마지막 페이지에서만 실제 WRITE 발행) */
	if (cpp->enable_gc_delay) {
		struct nand_cmd gcw = {
			.type = type,
			.cmd = NAND_NOP,
			.stime = 0,
			.interleave_pci_dma = false,
			.ppa = &new_ppa,
		};
		if (last_pg_in_wordline(conv_ftl, &new_ppa)) {
			gcw.cmd = NAND_WRITE;
			gcw.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg;
		}
		ssd_advance_nand(conv_ftl->ssd, &gcw);
	}

	/* RR은 write credit을 소모한다 (쓰기 대역폭 공유) */
	if (type == RR_IO) {
		consume_write_credit(conv_ftl);
	}

	return 0;
}

/*
 * select_victim_line: GC를 위한 victim line(슈퍼블록) 선택.
 * vpc가 가장 낮은 line을 우선순위 큐에서 꺼낸다.
 *
 * force == false: vpc가 전체의 1/8 이하인 line만 선택 (효율적 GC)
 * force == true:  어떤 line이든 선택 (긴급 GC, free line 부족 시)
 */
static struct line *select_victim_line(struct conv_ftl *conv_ftl, bool force)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *victim_line = NULL;

	victim_line = pqueue_peek(lm->victim_line_pq);
	if (!victim_line) {
		return NULL;
	}

	if (!force && (victim_line->vpc > (spp->pgs_per_line / 8))) {
		return NULL;
	}

	pqueue_pop(lm->victim_line_pq);
	victim_line->pos = 0;
	lm->victim_line_cnt--;

	return victim_line;
}

/*
 * clean_one_flashpg: GC 시 하나의 flash page(one-shot program 단위 내
 * 여러 sub-page)를 읽고 valid 페이지를 새 위치로 복사한다.
 */
static void clean_one_flashpg(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	struct nand_page *pg_iter = NULL;
	int cnt = 0, i = 0;
	uint64_t completed_time = 0;
	struct ppa ppa_copy = *ppa;

	/* valid 페이지 수 카운트 */
	for (i = 0; i < spp->pgs_per_flashpg; i++) {
		pg_iter = get_pg(conv_ftl->ssd, &ppa_copy);
		NVMEV_ASSERT(pg_iter->status != PG_FREE);  /* victim block에 free page 없어야 함 */
		if (pg_iter->status == PG_VALID)
			cnt++;
		ppa_copy.g.pg++;
	}

	ppa_copy = *ppa;

	if (cnt <= 0)
		return;  /* valid 페이지 없으면 복사할 것 없음 */

	/* GC read: valid 페이지들을 DRAM으로 읽기 */
	if (cpp->enable_gc_delay) {
		struct nand_cmd gcr = {
			.type = GC_IO,
			.cmd = NAND_READ,
			.stime = 0,
			.xfer_size = spp->pgsz * cnt,
			.interleave_pci_dma = false,
			.ppa = &ppa_copy,
		};
		completed_time = ssd_advance_nand(conv_ftl->ssd, &gcr);
	}

	/* valid 페이지를 새 위치로 복사 */
	for (i = 0; i < spp->pgs_per_flashpg; i++) {
		pg_iter = get_pg(conv_ftl->ssd, &ppa_copy);
		if (pg_iter->status == PG_VALID) {
			gc_write_page(conv_ftl, &ppa_copy, GC_IO);
		}
		ppa_copy.g.pg++;
	}
}

/* line을 free 상태로 전환하고 free list에 추가 */
static void mark_line_free(struct conv_ftl *conv_ftl, struct ppa *ppa)
{
	struct line_mgmt *lm = &conv_ftl->lm;
	struct line *line = get_line(conv_ftl, ppa);
	line->ipc = 0;
	line->vpc = 0;
	list_add_tail(&line->entry, &lm->free_line_list);
	lm->free_line_cnt++;
	lm->tt_erase_cnt++;
}

/*
 * do_gc: GC를 수행한다.
 * victim line을 선택하고, 모든 채널/LUN의 블록에서 valid 페이지를 복사한 후,
 * 블록을 erase하고 line을 free로 전환한다.
 */
static int do_gc(struct conv_ftl *conv_ftl, bool force)
{
	struct line *victim_line = NULL;
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct ppa ppa;
	int flashpg;

	victim_line = select_victim_line(conv_ftl, force);
	if (!victim_line) {
		return -1;
	}

	ppa.g.blk = victim_line->id;
	NVMEV_DEBUG_VERBOSE("GC-ing line:%d,ipc=%d(%d),victim=%d,full=%d,free=%d\n",
		ppa.g.blk, victim_line->ipc, victim_line->vpc,
		conv_ftl->lm.victim_line_cnt, conv_ftl->lm.full_line_cnt,
		conv_ftl->lm.free_line_cnt);

	conv_ftl->wfc.credits_to_refill = victim_line->ipc;

	/* 블록 내 모든 flash page를 순회하며 valid 페이지 복사 */
	for (flashpg = 0; flashpg < spp->flashpgs_per_blk; flashpg++) {
		int ch, lun;
		ppa.g.pg = flashpg * spp->pgs_per_flashpg;

		for (ch = 0; ch < spp->nchs; ch++) {
			for (lun = 0; lun < spp->luns_per_ch; lun++) {
				struct nand_lun *lunp;
				ppa.g.ch = ch;
				ppa.g.lun = lun;
				ppa.g.pl = 0;
				lunp = get_lun(conv_ftl->ssd, &ppa);

				clean_one_flashpg(conv_ftl, &ppa);

				/* 마지막 flash page 처리 후 블록 erase */
				if (flashpg == (spp->flashpgs_per_blk - 1)) {
					struct convparams *cpp = &conv_ftl->cp;

					mark_block_free(conv_ftl, &ppa);

					if (cpp->enable_gc_delay) {
						struct nand_cmd gce = {
							.type = GC_IO,
							.cmd = NAND_ERASE,
							.stime = 0,
							.interleave_pci_dma = false,
							.ppa = &ppa,
						};
						ssd_advance_nand(conv_ftl->ssd, &gce);
					}

					lunp->gc_endtime = lunp->next_lun_avail_time;
				}
			}
		}
	}

	mark_line_free(conv_ftl, &ppa);

	return 0;
}

static void foreground_gc(struct conv_ftl *conv_ftl)
{
	if (should_gc_high(conv_ftl)) {
		NVMEV_DEBUG_VERBOSE("should_gc_high passed");
		do_gc(conv_ftl, true);
	}
}

/* 두 PPA가 같은 flash page에 속하는지 확인 (읽기 aggregation용) */
static bool is_same_flash_page(struct conv_ftl *conv_ftl, struct ppa ppa1, struct ppa ppa2)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	uint32_t ppa1_page = ppa1.g.pg / spp->pgs_per_flashpg;
	uint32_t ppa2_page = ppa2.g.pg / spp->pgs_per_flashpg;

	return (ppa1.h.blk_in_ssd == ppa2.h.blk_in_ssd) && (ppa1_page == ppa2_page);
}

/* ============================================================================
 * [IH Start] STRAW RR (Read Reclaim) 핵심 구현
 * ============================================================================ */

/*
 * clean_one_flashpg_for_rr: RR(Read Reclaim) 시 하나의 워드라인 내
 * 특정 flash page를 읽고 valid 페이지를 새 위치로 복사한다.
 *
 * GC의 clean_one_flashpg()와 다른 점:
 *   1) 워드라인 단위로 동작 (wl->first_ppa 기준)
 *   2) type = RR_IO로 gc_write_page() 호출
 *      → old page가 즉시 invalid 마킹됨 (WR²의 핵심)
 *   3) 빈(free) 페이지를 만나면 조기 종료 가능
 *      → 아직 프로그램되지 않은 WL의 남은 부분은 건너뜀
 *
 * 반환값:
 *   1 → 이 워드라인(또는 블록)의 RR이 완료되었음
 *   0 → 아직 더 처리할 flash page가 남아있음
 */
static int clean_one_flashpg_for_rr(struct conv_ftl *conv_ftl, struct wordline *wl, int flashidx)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	struct nand_page *pg_iter = NULL;

	int cnt = 0, i = 0, j = 0;
	uint64_t completed_time = 0;
	struct ppa ppa_copy = wl->first_ppa;
	/* 이 WL 내에서 flashidx번째 flash page의 시작 위치로 이동 */
	ppa_copy.g.pg = wl->first_ppa.g.pg + flashidx * spp->pgs_per_flashpg;

	/* valid 페이지 수 카운트 (free 페이지를 만나면 조기 종료) */
	for (i = 0; i < spp->pgs_per_flashpg; i++, j++) {
		pg_iter = get_pg(conv_ftl->ssd, &ppa_copy);
		ppa_copy.g.pg++;

		if (pg_iter->status == PG_FREE)
			break;  /* 아직 프로그램되지 않은 영역 → 종료 */
		else if (pg_iter->status == PG_VALID)
			cnt++;
	}

	if (cnt <= 0)
		/* valid 페이지 없음 → WL 종료 조건 확인 후 반환 */
		return (j < spp->pgs_per_flashpg) || (ppa_copy.g.pg % spp->pgs_per_blk == 0);

	/* 시작 위치로 되돌리기 */
	ppa_copy.g.pg = wl->first_ppa.g.pg + flashidx * spp->pgs_per_flashpg;

	/* RR read: valid 페이지를 DRAM으로 읽기 */
	if (cpp->enable_gc_delay) {
		struct nand_cmd gcr = {
			.type = RR_IO,
			.cmd = NAND_READ,
			.stime = 0,
			.xfer_size = spp->pgsz * cnt,
			.interleave_pci_dma = false,
			.ppa = &ppa_copy,
		};
		completed_time = ssd_advance_nand(conv_ftl->ssd, &gcr);
	}

	/* valid 페이지를 새 위치로 복사 (RR_IO 타입) */
	for (i = 0; i < spp->pgs_per_flashpg; i++) {
		pg_iter = get_pg(conv_ftl->ssd, &ppa_copy);
		if (pg_iter->status == PG_VALID) {
			gc_write_page(conv_ftl, &ppa_copy, RR_IO);
			wl->rr_pg_cnt++;  /* 이 WL에서 RR로 복사된 페이지 수 */
		}
		ppa_copy.g.pg++;
	}
	return (j < spp->pgs_per_flashpg) || (ppa_copy.g.pg % spp->pgs_per_blk == 0);
}

/*
 * get_rr_th: WL 그룹별 ERCMAX 임계값을 랜덤 변동 포함하여 반환.
 *
 * 논문 Figure 11의 ERCMAX를 기반으로, chip-to-chip variation을 모사하기 위해
 * [min_rr_th, max_rr_th] 범위 내에서 유사-랜덤 값을 생성한다.
 *
 * random_base와 random_add를 이용한 간단한 LCG(Linear Congruential Generator)
 * 방식으로 매 호출마다 다른 임계값을 반환한다.
 */
static uint32_t get_rr_th(struct conv_ftl *conv_ftl, int group_num) {
	struct wordline_mgmt *wlm = &conv_ftl->wlm;
	uint32_t scale = wlm->max_rr_th[group_num] - wlm->min_rr_th[group_num];
	wlm->random_base += wlm->random_add;

	return (wlm->random_base % scale) + wlm->min_rr_th[group_num];
}

/*
 * clean_ss: SS(Space-Saving) 모드에서 RR 완료 후 카운터를 정리한다.
 *
 * 인접 WL이 모두 invalid인 WL의 SS 카운터 엔트리를 무효화(invalidate).
 * 이유: 인접 WL이 invalid이면 해당 WL에 대한 read disturbance가 감소하므로,
 * 해당 WL의 read count 추적이 더 이상 필요하지 않다.
 *
 * 이는 SS 모드에서만 사용되며, EFFECT 모드에서는 eff_read_cnt로 대체된다.
 */
static void clean_ss(struct conv_ftl *conv_ftl, struct blk_wordline *bwl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	int oneshotpg;
	for (oneshotpg = 0; oneshotpg < spp->oneshotpgs_per_blk; oneshotpg++) {
		if (oneshotpg == 0) {
			/* 첫 번째 WL: 아래쪽 인접 WL만 확인 */
			if (is_wl_invalid(&bwl->wordlines[oneshotpg + 1]) == true)
				invalidate(bwl->counter, oneshotpg);
		}
		else if (oneshotpg == spp->oneshotpgs_per_blk - 1) {
			/* 마지막 WL: 위쪽 인접 WL만 확인 */
			if (is_wl_invalid(&bwl->wordlines[oneshotpg - 1]) == true)
				invalidate(bwl->counter, oneshotpg);
		}
		else {
			/* 중간 WL: 양쪽 인접 WL 모두 invalid이면 무효화 */
			if (is_wl_invalid(&bwl->wordlines[oneshotpg - 1]) == true &&
			    is_wl_invalid(&bwl->wordlines[oneshotpg + 1]) == true)
				invalidate(bwl->counter, oneshotpg);
		}
	}
}

/* ============================================================================
 * should_reclaim_wl_*: RR 모드별 WL reclaim 판단 함수들
 * ============================================================================
 * 각 함수는 현재 검사 중인 워드라인이 "heavily disturbed"인지 판단하여
 * reclaim 여부를 결정한다.
 *
 * 논문 Figure 16의 "WLi is heavily disturbed?" 판단에 해당.
 * ============================================================================ */

/*
 * should_reclaim_wl_pagetype: Pagetype 모드 (Han et al., TVLSI'23)
 *
 * 블록 내 페이지 위치(flash_seq)에 따라 다른 임계값을 적용:
 *   - 앞쪽 1/3: SRT (가장 낮은 임계값, MSB 페이지처럼 취약)
 *   - 중간 1/3: SRT × 1.5
 *   - 뒤쪽 1/3: SRT × 2.0 (가장 높은 임계값, LSB 페이지처럼 강건)
 *
 * 논문 §7.1: "Pagetype addresses heterogeneous impact across page types
 *   by setting page-type-level RR thresholds"
 */
static bool should_reclaim_wl_pagetype(struct conv_ftl *conv_ftl, struct blk_wordline* bwl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	int flashpg = bwl->flash_seq;

	int ratio1 = spp->flashpgs_per_blk / 3;
	int ratio2 = 2 * spp->flashpgs_per_blk / 3;

	if (flashpg <= ratio1)
		return bwl->read_cnt > SRT;
	else if (flashpg <= ratio2)
		return bwl->read_cnt > SRT * 1.5;
	else
		return bwl->read_cnt > SRT * 2;
}

/*
 * should_reclaim_wl_ss: SS(Space-Saving) 모드 — WR²의 핵심 (SR² 미포함).
 *
 * ===== 논문 대응: §6 Figure 16 — Identifying Heavily Disturbed WLs =====
 *
 * total_stress 계산:
 *   1) SS 카운터에서 현재 WL의 read count(wl_read_cnt)를 조회
 *   2) 인접 WL의 read count(adj_read_cnt)를 SS 카운터에서 조회
 *   3) total_stress = adj_read_cnt × α + (blk_read_cnt - adj_read_cnt - wl_read_cnt)
 *
 *   여기서:
 *   - adj_read_cnt × α: 인접 WL 읽기의 disturbance (α배 더 큰 stress)
 *     → 논문 §5.2: "reading adjacent WL causes 8.4× more disturbance"
 *   - (blk_read_cnt - adj_read_cnt - wl_read_cnt): 비인접 WL 읽기의 disturbance
 *     → 자기 자신(wl_read_cnt)과 인접 WL(adj_read_cnt) 읽기를 제외한 나머지
 *
 *   코드에서 α = 10으로 고정 (adj_read_cnt × 10), 논문의 α ≈ 8~10에 대응.
 *
 * ERCMAX와 비교하여 heavily disturbed 여부 판단:
 *   total_stress > get_rr_th(0) → reclaim 필요
 */
static bool should_reclaim_wl_ss(struct conv_ftl *conv_ftl, struct blk_wordline* bwl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	int flashpg = bwl->flash_seq;
	int oneshotpg = flashpg / spp->flashpgs_per_oneshotpg;

	uint32_t blk_read_cnt = bwl->read_cnt;    /* RC[BLK] */
	uint32_t wl_read_cnt = 0;                  /* RC[WL] (SS 카운터에서 근사) */
	uint32_t adj_read_cnt = 0;                 /* RC[adj] (인접 WL들의 read count 합) */
	uint32_t total_stress = 0;                 /* ERC[WL] (총 stress) */

	/*
	 * SS 카운터에서 현재 WL의 read count 조회.
	 * get(counter, 9999)는 SS 카운터의 최소값을 반환 (존재하지 않는 인덱스).
	 * wl_read_cnt > 최소값이면 SS 카운터에 엔트리가 있다는 의미.
	 *
	 * 주의: 초기값 wl_read_cnt=0이므로, 0 > get(counter, 9999)는 항상 false.
	 * 따라서 실제로는 SS 카운터에 엔트리가 없으면 wl_read_cnt=0으로 유지됨.
	 * (SS 알고리즘의 특성: 엔트리가 없으면 최소값 이하의 read count를 가짐)
	 */
	if (wl_read_cnt > get(bwl->counter, 9999))
		wl_read_cnt = get(bwl->counter, oneshotpg);

	/*
	 * 인접 WL의 read count 합산 (논문 Figure 16의 ②번 단계).
	 * 첫 번째/마지막 WL은 한쪽 인접 WL만 존재.
	 */
	if (oneshotpg == 0) {
		adj_read_cnt = get(bwl->counter, oneshotpg + 1);
	}
	else if (oneshotpg == spp->oneshotpgs_per_blk - 1) {
		adj_read_cnt = get(bwl->counter, oneshotpg - 1);
	}
	else {
		adj_read_cnt = get(bwl->counter, oneshotpg - 1) +
		               get(bwl->counter, oneshotpg + 1);
	}

	/*
	 * ERC 계산 (논문 Figure 16의 ④번 단계):
	 *   ERC[WLi] = α × RC[adj] + RC[non-adj]
	 * 여기서:
	 *   α = 10 (하드코딩, 논문의 α ≈ 8~10)
	 *   RC[adj] = adj_read_cnt
	 *   RC[non-adj] = blk_read_cnt - adj_read_cnt - wl_read_cnt
	 *
	 * 전개하면:
	 *   total_stress = 10 × adj_read_cnt + (blk_read_cnt - adj_read_cnt - wl_read_cnt)
	 */
	total_stress = adj_read_cnt * 10 + (blk_read_cnt - adj_read_cnt - wl_read_cnt);

	/* 논문 Figure 16의 ⑤번 단계: ERC > ERCMAX이면 reclaim */
	return (total_stress > get_rr_th(conv_ftl, 0));
}

/*
 * should_reclaim_wl_oracle: Oracle 모드 — 이상적인 per-WL 카운터 사용.
 *
 * SS 카운터 대신 실제 per-WL read count(wl->read_cnt)를 사용한다.
 * SS 카운터의 근사 오차 없이 정확한 판단이 가능하므로
 * WR²의 이론적 상한(upper bound)을 평가하는 데 사용된다.
 *
 * 논문 Figure 19에서 "unconstrained REC"에 대응.
 */
static bool should_reclaim_wl_oracle(struct conv_ftl *conv_ftl, struct blk_wordline* bwl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	int flashpg = bwl->flash_seq;
	int oneshotpg = flashpg / spp->flashpgs_per_oneshotpg;

	uint32_t blk_read_cnt = bwl->read_cnt;
	uint32_t wl_read_cnt = bwl->wordlines[oneshotpg].read_cnt;  /* 실제 per-WL RC */
	uint32_t adj_read_cnt = 0;
	uint32_t total_stress = 0;

	if (oneshotpg == 0) {
		adj_read_cnt = bwl->wordlines[oneshotpg + 1].read_cnt;
	}
	else if (oneshotpg == spp->oneshotpgs_per_blk - 1) {
		adj_read_cnt = bwl->wordlines[oneshotpg - 1].read_cnt;
	}
	else {
		adj_read_cnt = bwl->wordlines[oneshotpg - 1].read_cnt +
		               bwl->wordlines[oneshotpg + 1].read_cnt;
	}

	total_stress = adj_read_cnt * 10 + (blk_read_cnt - adj_read_cnt - wl_read_cnt);

	return (total_stress > get_rr_th(conv_ftl, 0));
}

/*
 * should_reclaim_wl_ss_effective: STRAW의 핵심 — WR² + SR² 통합 판단.
 *
 * ===== 논문 대응: STRAW (WR² + SR²) =====
 *
 * should_reclaim_wl_ss()와의 차이점:
 *   1) blk_read_cnt 대신 eff_read_cnt/10을 사용
 *      → SR²에 의한 stress 감소가 반영된 effective read count
 *      → eff_read_cnt는 update_eff_rc()에서 [0,10] 범위의 stress를 누적
 *      → /10으로 나누면 "effective number of full-stress reads"가 됨
 *
 *   2) group_num 파라미터로 WL 그룹별 다른 ERCMAX 사용
 *      → get_rr_th(conv_ftl, group_num)
 *      → 논문 Figure 11: Best/Good/Bad/Worst 그룹별 ERCMAX
 *
 * 이 함수는 EFFECT 및 COCKTAIL_EFFECT 모드에서 사용된다.
 */
static bool should_reclaim_wl_ss_effective(struct conv_ftl *conv_ftl,
	struct blk_wordline* bwl, int group_num)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	int flashpg = bwl->flash_seq;
	int oneshotpg = flashpg / spp->flashpgs_per_oneshotpg;

	/*
	 * SR²의 효과 반영: eff_read_cnt / 10을 블록-레벨 RC로 사용.
	 * eff_read_cnt는 매 read마다 [0,10] 범위의 effective stress를 누적하므로,
	 * /10으로 나누면 "default Vpass 기준의 등가 read 횟수"가 된다.
	 *
	 * 예: 100번 읽었는데 매번 stress가 7이었다면,
	 *     eff_read_cnt = 700, blk_read_cnt = 700/10 = 70
	 *     → default Vpass로 70번 읽은 것과 동등한 stress
	 */
	uint32_t blk_read_cnt = bwl->eff_read_cnt / 10;
	uint32_t wl_read_cnt = 0;
	uint32_t adj_read_cnt = 0;
	uint32_t total_stress = 0;

	if (wl_read_cnt > get(bwl->counter, 9999))
		wl_read_cnt = get(bwl->counter, oneshotpg);

	if (oneshotpg == 0) {
		adj_read_cnt = get(bwl->counter, oneshotpg + 1);
	}
	else if (oneshotpg == spp->oneshotpgs_per_blk - 1) {
		adj_read_cnt = get(bwl->counter, oneshotpg - 1);
	}
	else {
		adj_read_cnt = get(bwl->counter, oneshotpg - 1) +
		               get(bwl->counter, oneshotpg + 1);
	}

	total_stress = adj_read_cnt * 10 + (blk_read_cnt - adj_read_cnt - wl_read_cnt);

	/* WL 그룹별 ERCMAX와 비교 */
	return (total_stress > get_rr_th(conv_ftl, group_num));
}

/*
 * should_reclaim_wl_ss_effective_with_pagetype: STRAW + Pagetype 결합.
 *
 * should_reclaim_wl_ss_effective()에 Pagetype의 위치별 가중치를 추가:
 *   - 블록 앞쪽 1/3: rr_th 그대로 (가장 취약)
 *   - 블록 중간 1/3: rr_th × 1.5
 *   - 블록 뒤쪽 1/3: rr_th × 2.0 (가장 강건)
 *
 * PAGETYPE_EFFECT 모드에서 사용.
 */
static bool should_reclaim_wl_ss_effective_with_pagetype(struct conv_ftl *conv_ftl,
	struct blk_wordline* bwl, int group_num)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	int flashpg = bwl->flash_seq;
	int oneshotpg = flashpg / spp->flashpgs_per_oneshotpg;
	uint32_t rr_th = get_rr_th(conv_ftl, group_num);

	uint32_t blk_read_cnt = bwl->eff_read_cnt / 10;
	uint32_t wl_read_cnt = 0;
	uint32_t adj_read_cnt = 0;
	uint32_t total_stress = 0;

	int ratio1 = spp->flashpgs_per_blk / 3;
	int ratio2 = 2 * spp->flashpgs_per_blk / 3;

	if (wl_read_cnt > get(bwl->counter, 9999))
		wl_read_cnt = get(bwl->counter, oneshotpg);

	if (oneshotpg == 0) {
		adj_read_cnt = get(bwl->counter, oneshotpg + 1);
	}
	else if (oneshotpg == spp->oneshotpgs_per_blk - 1) {
		adj_read_cnt = get(bwl->counter, oneshotpg - 1);
	}
	else {
		adj_read_cnt = get(bwl->counter, oneshotpg - 1) +
		               get(bwl->counter, oneshotpg + 1);
	}

	total_stress = adj_read_cnt * 10 + (blk_read_cnt - adj_read_cnt - wl_read_cnt);

	/* Pagetype 가중치 적용 */
	if (flashpg <= ratio1)
		return total_stress > rr_th;            /* 앞쪽: 기본 임계값 */
	else if (flashpg <= ratio2)
		return total_stress > (3 * rr_th) / 2;  /* 중간: 1.5× */
	else
		return total_stress > rr_th * 2;        /* 뒤쪽: 2.0× */
}

/*
 * do_rr: 특정 블록(bwl)에 대해 RR을 수행한다.
 *
 * ===== 논문 대응: Figure 15 "For every valid WLi in BLKtgt" 루프 =====
 *
 * flash_seq를 사용하여 블록 내 WL을 순차적으로 검사한다.
 * flash_seq는 현재 검사 중인 flash page 인덱스를 나타내며,
 * 매 호출마다 1씩 증가한다.
 *
 * RR 모드별 동작:
 *
 * BLOCK:
 *   블록 전체 RC가 SRT(Simple Read Threshold)를 초과하면
 *   모든 WL의 valid 페이지를 순차적으로 복사.
 *   (기존 Baseline 방식 — 논문의 비교 대상)
 *
 * PAGETYPE:
 *   위치별 임계값을 적용한 블록-레벨 RR.
 *
 * COCKTAIL:
 *   BLOCK과 동일한 RR이지만, check_rr에서 랜덤 블록을 대상으로
 *   트리거하여 핫 페이지를 분산시킴.
 *
 * SS (Space-Saving):
 *   WL-level RR (WR² only, SR² 미포함).
 *   should_reclaim_wl_ss()로 heavily disturbed WL만 선별 reclaim.
 *   → reclaim 불필요한 WL은 flash_seq만 증가시키고 건너뜀.
 *
 * ORACLE:
 *   정확한 per-WL RC를 사용한 WL-level RR (이론적 상한).
 *
 * EFFECT / COCKTAIL_EFFECT:
 *   STRAW 전체 (WR² + SR²).
 *   should_reclaim_wl_ss_effective()로 effective RC 기반 판단.
 *   WL 그룹별(group_id) 다른 ERCMAX 적용.
 *
 * PAGETYPE_EFFECT:
 *   STRAW + Pagetype 결합.
 *
 * 반환값:
 *   1 → 이 블록의 RR이 완료됨 (모든 WL 검사 완료 또는 블록 끝 도달)
 *   0 → 아직 더 검사할 WL이 남아있음 (다음 호출에서 계속)
 */
static int do_rr(struct conv_ftl *conv_ftl, struct blk_wordline* bwl)
{
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	int flashpg = bwl->flash_seq;
	int oneshotpg = flashpg / spp->flashpgs_per_oneshotpg;
	int flashidx = flashpg % spp->flashpgs_per_oneshotpg;
	struct wordline *wl = &bwl->wordlines[oneshotpg];

	switch (cpp->rr_mode) {
	case BLOCK:
		/* 블록-레벨 RR: RC > SRT이면 모든 페이지 복사 */
		if (bwl->read_cnt > SRT) {
			int res = clean_one_flashpg_for_rr(conv_ftl, wl, flashidx);
			++bwl->flash_seq;
			return res;
		}
		return 1;

	case PAGETYPE:
		/* Pagetype RR: 위치별 다른 임계값 */
		if (should_reclaim_wl_pagetype(conv_ftl, bwl)) {
			int res = clean_one_flashpg_for_rr(conv_ftl, wl, flashidx);
			++bwl->flash_seq;
			return res;
		}
		return 1;

	case COCKTAIL:
		/* Cocktail: BLOCK과 동일하지만 랜덤 블록 대상 */
		if (bwl->read_cnt > SRT) {
			int res = clean_one_flashpg_for_rr(conv_ftl, wl, flashidx);
			++bwl->flash_seq;
			return res;
		}
		return 1;

	case SS:
		/* WR² (SS 카운터 기반): heavily disturbed WL만 reclaim */
		if (should_reclaim_wl_ss(conv_ftl, bwl)) {
			int res = clean_one_flashpg_for_rr(conv_ftl, wl, flashidx);
			++bwl->flash_seq;
			return res;
		}
		/* reclaim 불필요 → 다음 WL로 건너뜀 */
		return ++bwl->flash_seq == spp->flashpgs_per_blk;

	case ORACLE:
		/* Oracle: 정확한 per-WL RC로 판단 */
		if (should_reclaim_wl_oracle(conv_ftl, bwl)) {
			int res = clean_one_flashpg_for_rr(conv_ftl, wl, flashidx);
			++bwl->flash_seq;
			return res;
		}
		return ++bwl->flash_seq == spp->flashpgs_per_blk;

	case EFFECT:
	case COCKTAIL_EFFECT:
		/* STRAW (WR² + SR²): effective RC + WL 그룹별 ERCMAX */
		if (should_reclaim_wl_ss_effective(conv_ftl, bwl, wl->group_id)) {
			int res = clean_one_flashpg_for_rr(conv_ftl, wl, flashidx);
			++bwl->flash_seq;
			return res;
		}
		return ++bwl->flash_seq == spp->flashpgs_per_blk;

	case PAGETYPE_EFFECT:
		/* STRAW + Pagetype */
		if (should_reclaim_wl_ss_effective_with_pagetype(conv_ftl, bwl, wl->group_id)) {
			int res = clean_one_flashpg_for_rr(conv_ftl, wl, flashidx);
			++bwl->flash_seq;
			return res;
		}
		/* 주의: 여기에 return 누락 — fall-through로 1 반환 */
	}

	return 1;
}

/*
 * rr_procedure: 매 읽기 I/O 후 호출되어 RR victim 리스트의 모든 블록을 처리한다.
 *
 * ===== 논문 대응: Figure 15 전체 흐름 =====
 *
 * 동작:
 *   1) RR victim이 없으면 아무것도 하지 않음
 *   2) 모든 채널/LUN의 RR victim 리스트를 순회
 *   3) 각 victim 블록에 대해 do_rr() 호출
 *   4) do_rr()이 1을 반환하면 (해당 블록의 RR 완료):
 *      - SS 모드: clean_ss()로 SS 카운터 정리
 *      - victim 리스트에서 제거, is_rr = false
 *   5) 모든 victim이 처리될 때까지 반복
 *   6) 완료 후 write credit 확인 및 보충
 *
 * 이 함수는 conv_read()의 마지막에서 호출되므로,
 * 매 읽기 요청 처리 후 pending된 RR 작업을 수행한다.
 */
static void rr_procedure(struct nvmev_ns *ns)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *conv_ftl = &conv_ftls[0];

	struct wordline_mgmt *wlm = &conv_ftl->wlm;
	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct convparams *cpp = &conv_ftl->cp;
	int ch, lun;

	if (wlm->rr_victim_cnt > 0) {
		do {
			for (ch = 0; ch < spp->nchs; ch++) {
				for (lun = 0; lun < spp->luns_per_ch; lun++) {
					struct blk_wordline *bwl = list_first_entry_or_null(
						&wlm->rr_victim_wl_list[ch][lun],
						struct blk_wordline, entry);
					if (bwl) {
						if (do_rr(conv_ftl, bwl) == 1) {
							/* RR 완료: victim 리스트에서 제거 */
							if (cpp->rr_mode == SS)
								clean_ss(conv_ftl, bwl);
							list_del_init(&bwl->entry);
							bwl->is_rr = false;
							if (--wlm->rr_victim_cnt == 0) {
								/* 모든 victim 처리 완료 → 루프 탈출 */
								ch = spp->nchs;
								lun = spp->luns_per_ch;
							}
						}
					}
				}
			}
		} while (wlm->rr_victim_cnt > 0);

		/* RR로 인해 소모된 write credit 보충 */
		check_and_refill_write_credit(conv_ftl);
	}
}
/* [IH End] — RR 구현 */

/* ============================================================================
 * NVMe Read/Write/Flush 명령어 처리
 * ============================================================================ */

/*
 * conv_read: NVMe 읽기 명령을 처리한다.
 *
 * 1) LBA → LPN 변환 후 매핑 테이블에서 PPA 조회
 * 2) 같은 flash page에 속하는 연속 읽기를 aggregation (NAND 대역폭 최적화)
 * 3) NAND read latency 시뮬레이션
 * 4) [IH] check_rr(): 매 flash page 읽기 후 read disturbance 추적
 * 5) [IH] rr_procedure(): 모든 읽기 완료 후 pending RR 수행
 *
 * 논문에서 설명하는 read 흐름 (Figure 15):
 *   "Whenever a page is read from WLtgt in BLKtgt, StrawFTL first selects
 *    the minimum-safe Vpass mode, then updates the REC entries..."
 *   → check_rr()에서 이를 수행
 */
static bool conv_read(struct nvmev_ns *ns, struct nvmev_request *req, struct nvmev_result *ret)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *conv_ftl = &conv_ftls[0];
	struct ssdparams *spp = &conv_ftl->ssd->sp;

	struct nvme_command *cmd = req->cmd;
	uint64_t lba = cmd->rw.slba;
	uint64_t nr_lba = (cmd->rw.length + 1);
	uint64_t start_lpn = lba / spp->secs_per_pg;
	uint64_t end_lpn = (lba + nr_lba - 1) / spp->secs_per_pg;
	uint64_t lpn;
	uint64_t nsecs_start = req->nsecs_start;
	uint64_t nsecs_completed, nsecs_latest = nsecs_start;
	uint32_t xfer_size, i;
	uint32_t nr_parts = ns->nr_parts;

	struct ppa prev_ppa;
	struct nand_cmd srd = {
		.type = USER_IO,
		.cmd = NAND_READ,
		.stime = nsecs_start,
		.interleave_pci_dma = true,
	};

	NVMEV_ASSERT(conv_ftls);
	NVMEV_DEBUG_VERBOSE("%s: start_lpn=%lld, len=%lld, end_lpn=%lld",
		__func__, start_lpn, nr_lba, end_lpn);

	if ((end_lpn / nr_parts) >= spp->tt_pgs) {
		NVMEV_ERROR("%s: lpn passed FTL range (start_lpn=%lld > tt_pgs=%ld)\n",
			    __func__, start_lpn, spp->tt_pgs);
		return false;
	}

	/* 펌웨어 처리 지연 추가 (4KB 이하 vs 그 이상) */
	if (LBA_TO_BYTE(nr_lba) <= (KB(4) * nr_parts)) {
		srd.stime += spp->fw_4kb_rd_lat;
	} else {
		srd.stime += spp->fw_rd_lat;
	}

	for (i = 0; (i < nr_parts) && (start_lpn <= end_lpn); i++, start_lpn++) {
		conv_ftl = &conv_ftls[start_lpn % nr_parts];
		xfer_size = 0;
		prev_ppa = get_maptbl_ent(conv_ftl, start_lpn / nr_parts);

		/* 각 LPN에 대해 NAND 읽기 수행 */
		for (lpn = start_lpn; lpn <= end_lpn; lpn += nr_parts) {
			uint64_t local_lpn;
			struct ppa cur_ppa;

			local_lpn = lpn / nr_parts;
			cur_ppa = get_maptbl_ent(conv_ftl, local_lpn);
			if (!mapped_ppa(&cur_ppa) || !valid_ppa(conv_ftl, &cur_ppa)) {
				NVMEV_DEBUG_VERBOSE("lpn 0x%llx not mapped to valid ppa\n", local_lpn);
				continue;
			}

			conv_ftl->lm.tt_read_cnt++;  /* NAND read 횟수 통계 */

			/* 같은 flash page 내 연속 읽기 aggregation */
			if (mapped_ppa(&prev_ppa) &&
			    is_same_flash_page(conv_ftl, cur_ppa, prev_ppa)) {
				xfer_size += spp->pgsz;
				continue;
			}

			/* 이전 flash page의 읽기 발행 */
			if (xfer_size > 0) {
				srd.xfer_size = xfer_size;
				srd.ppa = &prev_ppa;
				nsecs_completed = ssd_advance_nand(conv_ftl->ssd, &srd);
				nsecs_latest = max(nsecs_completed, nsecs_latest);
				check_rr(conv_ftl, srd.ppa);  /* [IH] read disturbance 추적 */
			}

			xfer_size = spp->pgsz;
			prev_ppa = cur_ppa;
		}

		/* 마지막 flash page의 읽기 발행 */
		if (xfer_size > 0) {
			srd.xfer_size = xfer_size;
			srd.ppa = &prev_ppa;
			nsecs_completed = ssd_advance_nand(conv_ftl->ssd, &srd);
			nsecs_latest = max(nsecs_completed, nsecs_latest);
			check_rr(conv_ftl, srd.ppa);  /* [IH] read disturbance 추적 */
		}
	}

	ret->nsecs_target = nsecs_latest;
	ret->status = NVME_SC_SUCCESS;

	rr_procedure(ns);  /* [IH] pending RR 수행 */
	return true;
}

/*
 * conv_write: NVMe 쓰기 명령을 처리한다.
 *
 * 1) write buffer 할당 및 PCIe 전송 시뮬레이션
 * 2) 기존 매핑이 있으면 old page invalidation
 * 3) 새 페이지 할당 → 매핑 업데이트 → valid 마킹
 * 4) WL의 마지막 페이지에서 NAND WRITE 발행 (one-shot programming)
 * 5) write credit 소모 및 필요 시 GC 트리거
 */
static bool conv_write(struct nvmev_ns *ns, struct nvmev_request *req, struct nvmev_result *ret)
{
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *conv_ftl = &conv_ftls[0];

	struct ssdparams *spp = &conv_ftl->ssd->sp;
	struct buffer *wbuf = conv_ftl->ssd->write_buffer;

	struct nvme_command *cmd = req->cmd;
	uint64_t lba = cmd->rw.slba;
	uint64_t nr_lba = (cmd->rw.length + 1);
	uint64_t start_lpn = lba / spp->secs_per_pg;
	uint64_t end_lpn = (lba + nr_lba - 1) / spp->secs_per_pg;

	uint64_t lpn;
	uint32_t nr_parts = ns->nr_parts;

	uint64_t nsecs_latest;
	uint64_t nsecs_xfer_completed;
	uint32_t allocated_buf_size;

	struct nand_cmd swr = {
		.type = USER_IO,
		.cmd = NAND_WRITE,
		.interleave_pci_dma = false,
		.xfer_size = spp->pgsz * spp->pgs_per_oneshotpg,
	};

	NVMEV_DEBUG_VERBOSE("%s: start_lpn=%lld, len=%lld, end_lpn=%lld",
		__func__, start_lpn, nr_lba, end_lpn);

	if ((end_lpn / nr_parts) >= spp->tt_pgs) {
		NVMEV_ERROR("%s: lpn passed FTL range (start_lpn=%lld > tt_pgs=%ld)\n",
				__func__, start_lpn, spp->tt_pgs);
		return false;
	}

	/* write buffer 할당 */
	allocated_buf_size = buffer_allocate(wbuf, LBA_TO_BYTE(nr_lba));
	if (allocated_buf_size < LBA_TO_BYTE(nr_lba))
		return false;

	/* PCIe 전송 시뮬레이션 */
	nsecs_latest = ssd_advance_write_buffer(conv_ftl->ssd, req->nsecs_start, LBA_TO_BYTE(nr_lba));
	nsecs_xfer_completed = nsecs_latest;

	swr.stime = nsecs_latest;

	for (lpn = start_lpn; lpn <= end_lpn; lpn++) {
		uint64_t local_lpn;
		uint64_t nsecs_completed = 0;
		struct ppa ppa;

		conv_ftl = &conv_ftls[lpn % nr_parts];
		local_lpn = lpn / nr_parts;

		/* 기존 매핑 확인: overwrite이면 old page invalidation */
		ppa = get_maptbl_ent(conv_ftl, local_lpn);
		if (mapped_ppa(&ppa)) {
			mark_page_invalid(conv_ftl, &ppa);
			set_rmap_ent(conv_ftl, INVALID_LPN, &ppa);
			NVMEV_DEBUG("%s: %lld is invalid, ", __func__, ppa2pgidx(conv_ftl, &ppa));
		}

		/* 새 페이지 할당 */
		ppa = get_new_page(conv_ftl, USER_IO);
		set_maptbl_ent(conv_ftl, local_lpn, &ppa);
		NVMEV_DEBUG("%s: got new ppa %lld, ", __func__, ppa2pgidx(conv_ftl, &ppa));
		set_rmap_ent(conv_ftl, local_lpn, &ppa);
		mark_page_valid(conv_ftl, &ppa);

		advance_write_pointer(conv_ftl, USER_IO);

		/* WL의 마지막 페이지에서 NAND WRITE 발행 */
		if (last_pg_in_wordline(conv_ftl, &ppa)) {
			swr.ppa = &ppa;
			nsecs_completed = ssd_advance_nand(conv_ftl->ssd, &swr);
			nsecs_latest = max(nsecs_completed, nsecs_latest);

			schedule_internal_operation(req->sq_id, nsecs_completed, wbuf,
						    spp->pgs_per_oneshotpg * spp->pgsz);
		}

		consume_write_credit(conv_ftl);
		check_and_refill_write_credit(conv_ftl);

		conv_ftl->lm.tt_write_cnt++;
	}

	if ((cmd->rw.control & NVME_RW_FUA) || (spp->write_early_completion == 0)) {
		ret->nsecs_target = nsecs_latest;       /* 모든 NAND 연산 완료 대기 */
	} else {
		ret->nsecs_target = nsecs_xfer_completed; /* early completion */
	}
	ret->status = NVME_SC_SUCCESS;

	return true;
}

/* NVMe flush 명령: 모든 파티션의 NAND 연산이 완료될 때까지 대기 */
static void conv_flush(struct nvmev_ns *ns, struct nvmev_request *req, struct nvmev_result *ret)
{
	uint64_t start, latest;
	uint32_t i;
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;

	start = local_clock();
	latest = start;
	for (i = 0; i < ns->nr_parts; i++) {
		latest = max(latest, ssd_next_idle_time(conv_ftls[i].ssd));
	}

	NVMEV_DEBUG_VERBOSE("%s: latency=%llu\n", __func__, latest - start);

	ret->status = NVME_SC_SUCCESS;
	ret->nsecs_target = latest;
}

/*
 * conv_proc_nvme_io_cmd: NVMe I/O 명령어 디스패처.
 *
 * Read/Write/Flush 명령을 각 핸들러로 분기하고,
 * 매 50,000번의 read I/O마다 통계를 출력한다 (STRAW 평가용).
 */
bool conv_proc_nvme_io_cmd(struct nvmev_ns *ns, struct nvmev_request *req, struct nvmev_result *ret)
{
	struct nvme_command *cmd = req->cmd;
	struct conv_ftl *conv_ftls = (struct conv_ftl *)ns->ftls;
	struct conv_ftl *conv_ftl = &conv_ftls[0];

	NVMEV_ASSERT(ns->csi == NVME_CSI_NVM);
	switch (cmd->common.opcode) {
	case nvme_cmd_write:
		if (!conv_write(ns, req, ret))
			return false;
		conv_ftl->tt_write_io_cnt++;
		break;
	case nvme_cmd_read:
		if (!conv_read(ns, req, ret))
			return false;
		conv_ftl->tt_read_io_cnt++;

		/* 주기적 통계 출력 (50,000 read I/O마다) */
		if (conv_ftl->tt_read_io_cnt % 50000 == 0)
			print_statistic(conv_ftl);
		break;
	case nvme_cmd_flush:
		conv_flush(ns, req, ret);
		break;
	default:
		NVMEV_ERROR("%s: command not implemented: %s (0x%x)\n", __func__,
				nvme_opcode_string(cmd->common.opcode), cmd->common.opcode);
		break;
	}

	return true;
}
