// The rules a suite combines its members' predictions with, one per
// `combine` value of its manifest. Each takes the members' outputs, one array
// per member and output, and returns the target's columns.

const DEG = Math.PI / 180;

function std(values, mean) {
  if (values.length < 2) return NaN;
  let s = 0;
  for (const v of values) s += (v - mean) ** 2;
  return Math.sqrt(s / (values.length - 1));
}

// Classification: each member's most probable class is a vote; the class with
// the most votes wins, ties by the mean probability.
function vote(target, members, n) {
  const classes = target.schema.heads[0].classes;
  const c = classes.length;
  const label = new Array(n);
  const mean = classes.map(() => new Float64Array(n));
  const seedLabels = members.map(() => new Array(n));
  for (let i = 0; i < n; i += 1) {
    const votes = new Int32Array(c);
    const prob = new Float64Array(c);
    members.forEach(([probs], m) => {
      let best = 0;
      for (let k = 0; k < c; k += 1) {
        const p = probs[i * c + k];
        prob[k] += p / members.length;
        if (p > probs[i * c + best]) best = k;
      }
      votes[best] += 1;
      seedLabels[m][i] = classes[best];
    });
    const most = Math.max(...votes);
    let win = -1;
    for (let k = 0; k < c; k += 1) {
      if (votes[k] === most && (win < 0 || prob[k] > prob[win])) win = k;
      mean[k][i] = prob[k];
    }
    label[i] = classes[win];
  }
  const columns = [[target.name, label]];
  classes.forEach((cls, k) => columns.push([`${target.name}_prob_${cls}`, mean[k]]));
  return { columns, seeds: seedLabels };
}

// Regression in original units: the mean and the standard deviation over members.
function mean(target, members, n) {
  const mu = new Float64Array(n);
  const sd = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const values = members.map(([y]) => y[i]);
    mu[i] = values.reduce((a, b) => a + b, 0) / values.length;
    sd[i] = std(values, mu[i]);
  }
  return {
    columns: [[target.name, mu], [`${target.name}_sd`, sd]],
    seeds: members.map(([y]) => Float64Array.from(y)),
  };
}

// Coordinates: the members' points on the sphere, averaged as unit vectors.
function sphericalMean(target, members, n) {
  const lat = new Float64Array(n);
  const lon = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    let x = 0;
    let y = 0;
    let z = 0;
    for (const [la, lo] of members) {
      const phi = la[i] * DEG;
      const lambda = lo[i] * DEG;
      x += Math.cos(phi) * Math.cos(lambda);
      y += Math.cos(phi) * Math.sin(lambda);
      z += Math.sin(phi);
    }
    const r = Math.hypot(x, y, z);
    lat[i] = Math.atan2(z / r, Math.hypot(x, y) / r) / DEG;
    lon[i] = Math.atan2(y, x) / DEG;
  }
  return {
    columns: [[`${target.name}_latitude`, lat], [`${target.name}_longitude`, lon]],
    seeds: members.map(([la, lo]) => [Float64Array.from(la), Float64Array.from(lo)]),
  };
}

export const COMBINE = { vote, mean, spherical_mean: sphericalMean };

export function combine(target, members, n) {
  const rule = COMBINE[target.combine];
  if (!rule) throw new Error(`no combine rule '${target.combine}' for ${target.name}`);
  return rule(target, members, n);
}
