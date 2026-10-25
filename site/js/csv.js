// Delimited text to rows of strings (RFC 4180 quoting). The delimiter is the
// one of comma, semicolon and tab that occurs most often in the first line
// outside quotes.

const DELIMITERS = [",", ";", "\t"];

export function detectDelimiter(text) {
  const counts = new Map(DELIMITERS.map((d) => [d, 0]));
  let quoted = false;
  for (const ch of text) {
    if (ch === '"') quoted = !quoted;
    else if (!quoted && (ch === "\n" || ch === "\r")) break;
    else if (!quoted && counts.has(ch)) counts.set(ch, counts.get(ch) + 1);
  }
  let best = ",";
  for (const [d, n] of counts) if (n > counts.get(best)) best = d;
  return best;
}

export function parseDelimited(text, delimiter = detectDelimiter(text)) {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  let i = 0;
  const n = text.length;
  while (i < n) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 2;
          continue;
        }
        quoted = false;
      } else {
        field += ch;
      }
      i += 1;
      continue;
    }
    if (ch === '"' && field === "") {
      quoted = true;
    } else if (ch === delimiter) {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      row.push(field);
      field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
    } else {
      field += ch;
    }
    i += 1;
  }
  if (field !== "" || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

// Rows to a table: column names from the first row, columns as string arrays.
export function toTable(rows) {
  const [names, ...body] = rows;
  const columns = new Map(names.map((name) => [name.trim(), new Array(body.length)]));
  const keys = names.map((name) => name.trim());
  for (let r = 0; r < body.length; r += 1) {
    const cells = body[r];
    for (let c = 0; c < keys.length; c += 1) columns.get(keys[c])[r] = cells[c] ?? "";
  }
  return { names: keys, columns, nRows: body.length };
}

// A table from named columns of equal length.
export function tableFromColumns(columns) {
  const names = Object.keys(columns);
  return {
    names,
    columns: new Map(names.map((n) => [n, columns[n]])),
    nRows: names.length ? columns[names[0]].length : 0,
  };
}

export function readTable(text) {
  return toTable(parseDelimited(text));
}
