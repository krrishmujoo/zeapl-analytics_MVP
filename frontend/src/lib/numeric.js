// Numeric parsing/formatting used across the analysis workspace.
//
// parseNumericValue() is intentionally strict about what counts as "no
// value" (null/undefined/empty/"N/A"/non-numeric text) vs. a real number,
// including zero -- zero is a valid, meaningful value and must never be
// treated the same as "missing".

const CURRENCY_AND_WHITESPACE = /[₹$€£,%\s]/g;

export function parseNumericValue(value) {
  if (value === null || value === undefined) return null;

  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }

  if (typeof value !== "string") return null;

  const trimmed = value.trim();
  if (!trimmed) return null;

  const lowered = trimmed.toLowerCase();
  if (["n/a", "na", "null", "undefined", "-", "--"].includes(lowered)) {
    return null;
  }

  const cleaned = trimmed.replace(CURRENCY_AND_WHITESPACE, "");
  if (!cleaned || cleaned === "-" || cleaned === "." || cleaned === "-.") return null;

  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : null;
}

// Formats a number (or numeric-looking string) with thousands separators.
// Whole numbers render with no decimal places; fractional values keep up
// to 2. Returns an em dash for anything that doesn't parse, so callers
// never have to special-case "no data" themselves.
export function formatNumber(value, { decimals } = {}) {
  const n = parseNumericValue(value);
  if (n === null) return "—";

  if (typeof decimals === "number") {
    return n.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }

  const hasFraction = Math.abs(n % 1) > 1e-9;
  return n.toLocaleString("en-US", {
    minimumFractionDigits: hasFraction ? 2 : 0,
    maximumFractionDigits: 2,
  });
}

// "invoice_value" / "invoiceValue" / "Invoice Value" -> "Invoice Value"
export function titleCaseKey(key) {
  if (key === null || key === undefined || key === "") return "";
  const spaced = String(key)
    .replace(/[_\-]+/g, " ")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .trim();
  if (!spaced) return "";
  return spaced
    .split(/\s+/)
    .map((word) => (word.length ? word[0].toUpperCase() + word.slice(1).toLowerCase() : word))
    .join(" ");
}
