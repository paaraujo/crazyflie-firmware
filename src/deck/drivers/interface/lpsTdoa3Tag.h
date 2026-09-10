#ifndef __LPS_TDOA3_TAG_H__
#define __LPS_TDOA3_TAG_H__

#include <stdbool.h>
#include <stdint.h>

#include "locodeck.h"
#include "autoconf.h"

extern uwbAlgorithm_t uwbTdoa3TagAlgorithm;

#ifdef CONFIG_DECK_LOCO_TDOA3_HYBRID_MODE

/** Number of hybrid-mode TWR logging slots. Must match lpsTdoa3Tag.c. */
#define HYBRID_RANGE_SLOTS 6

/**
 * Read one hybrid-mode two-way-ranging slot.
 *
 * Intended for consumers that fuse the ranges themselves, such as the swarm
 * encirclement filter. Two-way ranging is asynchronous, and a sequential filter
 * must apply each measurement EXACTLY ONCE: re-applying one shrinks the
 * covariance without new evidence, and repeated often enough the gain collapses
 * and the filter stops responding to data while still reporting confidence.
 *
 * Polling a log variable cannot distinguish a fresh value from a stale one, so
 * this returns the update timestamp alongside the distance. Treat the sample as
 * new only when @p updated_ms differs from the last one consumed for that slot:
 *
 *     if (hybridGetRange(s, &id, &d, &t) && t != lastSeen[s] && d != 0.0f) {
 *         lastSeen[s] = t;
 *         scalarUpdate(id, d);
 *     }
 *
 * A distance of exactly 0.0 means the slot has not been refreshed within
 * HM_LOG_MAX_AGE_MS and carries no measurement; a real range is never zero.
 *
 * @param slot        Slot index, 0 .. HYBRID_RANGE_SLOTS-1.
 * @param remoteId    Out: the remote id this slot is configured for (hmLId<slot>).
 * @param distance    Out: last measured distance [m], or 0.0 if stale.
 * @param updated_ms  Out: system time of that measurement [ms].
 * @return false if @p slot is out of range, in which case no output is written.
 */
bool hybridGetRange(const uint8_t slot, uint8_t* remoteId, float* distance, uint32_t* updated_ms);

#endif // CONFIG_DECK_LOCO_TDOA3_HYBRID_MODE

#endif // __LPS_TDOA3_TAG_H__
