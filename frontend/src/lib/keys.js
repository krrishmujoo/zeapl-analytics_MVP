// Tolerant field lookup for objects whose keys might arrive in different
// cases/separators depending on backend version (snake_case, camelCase,
// "Title Case With Spaces", or prefixed like "total_invoice_value").
//
// This is a fallback layer, not the primary path: the v2 API contract in
// api.js is a clean, consistent snake_case schema, so most of the app reads
// known keys directly. pickField() exists for the genuinely variable bits
// (e.g. matching a chart's x_key against whatever column name a chart_data
// row actually used).

function canon(str) {
  return String(str).toLowerCase().replace(/[^a-z0-9]/g, "");
}

const AGG_PREFIX = /^(total|avg|average|sum|mean)/;

export function pickField(obj, candidates) {
  if (!obj || typeof obj !== "object" || !Array.isArray(candidates)) return undefined;

  // 1. Exact key match.
  for (const c of candidates) {
    if (c !== undefined && c !== null && Object.prototype.hasOwnProperty.call(obj, c)) {
      return obj[c];
    }
  }

  // 2. Case/separator-insensitive match.
  const canonMap = new Map();
  for (const k of Object.keys(obj)) {
    canonMap.set(canon(k), k);
  }
  for (const c of candidates) {
    if (c === undefined || c === null) continue;
    const hit = canonMap.get(canon(c));
    if (hit !== undefined) return obj[hit];
  }

  // 3. Same match ignoring a leading aggregation prefix on either side
  //    (so "invoice_value" can still find "total_invoice_value").
  for (const c of candidates) {
    if (c === undefined || c === null) continue;
    const target = canon(c).replace(AGG_PREFIX, "");
    if (!target) continue;
    for (const [ck, origKey] of canonMap.entries()) {
      if (ck.replace(AGG_PREFIX, "") === target) return obj[origKey];
    }
  }

  return undefined;
}
