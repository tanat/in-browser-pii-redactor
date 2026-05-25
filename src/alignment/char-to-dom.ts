/**
 * Map a character offset range in the plain-text content of `root` to a DOM Range.
 *
 * Plain-text content is the in-document-order concatenation of `Text` node values.
 * `start` and `end` are inclusive of the boundary at the start of a text node and
 * exclusive at its end (i.e., when an offset sits exactly on a boundary, prefer
 * the *next* text node for `start` and the *previous* text node for `end`, so that
 * the resulting Range hugs the actual highlight content rather than landing on a
 * sibling boundary). End offsets past the total length are clamped.
 */
export function charRangeToDomRange(
  root: HTMLElement,
  start: number,
  end: number,
): Range {
  if (end < start) {
    throw new Error(`charRangeToDomRange: end (${end}) < start (${start})`);
  }
  const doc = root.ownerDocument!;
  const range = doc.createRange();

  const walker = doc.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes: Text[] = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    nodes.push(n as Text);
  }

  if (nodes.length === 0) {
    range.setStart(root, root.childNodes.length);
    range.setEnd(root, root.childNodes.length);
    return range;
  }

  // Compute prefix lengths so node `i` covers chars [offsets[i], offsets[i+1]).
  const offsets = new Array<number>(nodes.length + 1);
  offsets[0] = 0;
  for (let i = 0; i < nodes.length; i++) {
    offsets[i + 1] = offsets[i] + nodes[i].data.length;
  }
  const total = offsets[nodes.length];

  const clampedStart = Math.max(0, Math.min(start, total));
  const clampedEnd = Math.max(clampedStart, Math.min(end, total));

  // Pick the start node: smallest i such that clampedStart < offsets[i+1], else last.
  // Tie-break on boundary: skip empty nodes; if non-empty, prefer the next one.
  let startIdx = nodes.length - 1;
  for (let i = 0; i < nodes.length; i++) {
    if (clampedStart < offsets[i + 1]) {
      startIdx = i;
      break;
    }
    if (clampedStart === offsets[i + 1]) {
      // Boundary: try to find a non-empty next node.
      let j = i + 1;
      while (j < nodes.length && nodes[j].data.length === 0) j++;
      startIdx = j < nodes.length ? j : i;
      break;
    }
  }

  // Pick the end node: largest i such that clampedEnd > offsets[i], else first.
  let endIdx = 0;
  for (let i = nodes.length - 1; i >= 0; i--) {
    if (clampedEnd > offsets[i]) {
      endIdx = i;
      break;
    }
  }

  range.setStart(nodes[startIdx], clampedStart - offsets[startIdx]);
  range.setEnd(nodes[endIdx], clampedEnd - offsets[endIdx]);
  return range;
}
