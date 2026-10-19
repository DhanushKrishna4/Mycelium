/** Recorded evidence only. Live simulation measurements never enter this view. */
const gates = ['growth', 'persistence', 'repair'];
const regimeNames = { growth: 'Growth only', persistence: 'Persistence', regeneration: 'Regeneration' };
const titleCase = (value) => String(value || '').replace(/[_-]+/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
const count = (value) => Number.isInteger(value) && value >= 0;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function modelLabel(model) {
  const target = model.target || model.metadata?.target;
  const regime = model.regime || model.metadata?.regime;
  if (target && regime) return `${titleCase(target)} · ${regimeNames[regime] || titleCase(regime)}`;
  return model.name || titleCase(model.id || model.model_id) || 'Unnamed checkpoint';
}

function aggregate(model, gate) {
  return model[gate] ?? model.aggregates?.[gate];
}

function reportLink(path) {
  if (typeof path !== 'string') return null;
  const relative = path.startsWith('reports/') ? path : `reports/${path}`;
  if (!/^reports\/[a-zA-Z0-9_./-]+$/.test(relative) || relative.split('/').some((part) => part === '..' || part === '.')) return null;
  return new URL(relative, import.meta.url).href;
}

function scoreCell(model, gate) {
  const cell = element('td');
  const score = element('div', 'result-score');
  const value = aggregate(model, gate);
  cell.dataset.label = titleCase(gate);
  cell.append(score);

  if (!value || typeof value !== 'object' || !count(value.successes) || !count(value.trials) || value.successes > value.trials) {
    score.dataset.outcome = 'unreported';
    score.append(element('strong', 'result-fraction', 'Not reported'), element('span', 'result-outcome', 'See the raw aggregate for available measurements.'));
    return cell;
  }
  if (value.trials === 0) {
    score.dataset.outcome = 'unreported';
    score.append(element('strong', 'result-fraction', 'No trials'), element('span', 'result-outcome', 'This gate was not evaluated.'));
    return cell;
  }

  score.dataset.outcome = value.successes === value.trials ? 'passed' : value.successes === 0 ? 'failed' : 'mixed';
  score.append(element('strong', 'result-fraction', `${value.successes} / ${value.trials}`));
  const meter = element('meter', 'result-meter');
  meter.min = 0;
  meter.max = value.trials;
  meter.value = value.successes;
  meter.setAttribute('aria-label', `${modelLabel(model)}, ${gate}: ${value.successes} of ${value.trials} trials successful`);
  meter.textContent = `${value.successes} of ${value.trials} successful`;
  score.append(meter);
  const failures = count(value.failures) ? value.failures : value.trials - value.successes;
  score.append(element('span', 'result-outcome', `${value.successes} successful · ${failures} failed`));

  if (gate === 'repair' && count(value.pre_damage_failures) && value.pre_damage_failures > 0) {
    score.append(element('span', 'result-note', `${value.pre_damage_failures} failed before damage`));
  }
  if (gate === 'repair' && count(value.retained_recoveries) && value.successes > 0) {
    score.append(element('span', 'result-note', `${value.retained_recoveries} / ${value.successes} recoveries retained`));
  }
  if (count(value.numerical_failures) && value.numerical_failures > 0) {
    score.append(element('span', 'result-note', `${value.numerical_failures} numerical ${value.numerical_failures === 1 ? 'failure' : 'failures'}`));
  }
  return cell;
}

function rawDetails(value, title, className = 'result-card') {
  const details = element('details', className);
  details.append(element('summary', '', title), element('pre', 'result-raw', JSON.stringify(value, null, 2)));
  return details;
}

/** Render the saved native evaluation without inferring unreported quality. */
export function renderResults(report, holder) {
  holder.replaceChildren();
  const models = Array.isArray(report?.models) ? report.models : report?.models && typeof report.models === 'object'
    ? Object.entries(report.models).map(([id, model]) => ({ id, ...model })) : Array.isArray(report?.results) ? report.results : [];
  const overview = element('section', 'results-overview');
  overview.setAttribute('aria-label', 'Evaluation scope');
  const facts = element('dl', 'results-facts');
  const fact = (label, value) => {
    const group = element('div');
    group.append(element('dt', '', label), element('dd', '', value));
    facts.append(group);
  };
  fact('Trained checkpoints', String(models.length));
  if (count(report?.total_trials)) fact('Recorded trials · including stress tests', String(report.total_trials));
  if (report?.config?.frozen_before_training === true) fact('Evaluation thresholds', 'Frozen before training');
  overview.append(facts);
  holder.append(overview);

  if (!models.length) {
    holder.append(element('p', 'results-warning', 'No model-level gate counts were reported. The raw report remains available below.'), rawDetails(report, 'Inspect the complete recorded report'));
    return;
  }

  holder.append(element('p', 'results-description', 'Successful trials / evaluated trials. Gates are measured separately; stress tests remain in the raw reports.'), element('p', 'results-scroll-hint', 'Scroll across to compare all three gates →'));
  const wrap = element('div', 'results-table-wrap');
  wrap.tabIndex = 0;
  wrap.setAttribute('role', 'region');
  wrap.setAttribute('aria-label', 'Recorded trial outcomes by model; scroll horizontally if needed');
  const table = element('table', 'results-table');
  table.append(element('caption', 'sr-only', 'Successful trials / evaluated trials for each trained rule.'));
  const head = element('thead');
  const headerRow = element('tr');
  const horizons = {
    growth: count(report?.config?.growth_horizon) ? `${report.config.growth_horizon} steps` : '',
    persistence: count(report?.config?.persistence_horizon) ? `${report.config.persistence_horizon} steps` : '',
    repair: count(report?.config?.repair_horizon) ? `${report.config.repair_horizon} steps after damage` : '',
  };
  for (const label of ['Trained rule', ...gates]) {
    const header = element('th');
    header.scope = 'col';
    header.append(element('span', 'result-column-title', titleCase(label)));
    if (horizons[label]) header.append(element('small', 'result-column-note', horizons[label]));
    headerRow.append(header);
  }
  head.append(headerRow);
  table.append(head);
  const body = element('tbody');
  for (const model of models) {
    const row = element('tr');
    const name = element('th', 'result-model');
    name.scope = 'row';
    name.append(element('strong', '', modelLabel(model)));
    const checkpoint = model.metadata?.checkpoint_update ?? model.checkpoint_update;
    if (count(checkpoint)) name.append(element('span', 'result-model-meta', `Checkpoint ${checkpoint.toLocaleString('en-US')}`));
    name.append(element('code', 'result-model-id', model.id || model.model_id || ''));
    row.append(name);
    for (const gate of gates) row.append(scoreCell(model, gate));
    body.append(row);
  }
  table.append(body);
  wrap.append(table);
  holder.append(wrap);

  if (report?.limitations) {
    const warning = element('section', 'results-warning');
    warning.setAttribute('aria-label', 'Limits of these findings');
    warning.append(element('h3', '', 'What these results can tell us'));
    const list = element('ul');
    for (const limitation of Array.isArray(report.limitations) ? report.limitations : [report.limitations]) list.append(element('li', '', String(limitation)));
    warning.append(list);
    holder.append(warning);
  }

  const sources = element('section', 'results-sources');
  sources.setAttribute('aria-label', 'Complete recorded evidence');
  sources.append(element('h3', '', 'Inspect the evidence'));
  sources.append(element('p', 'results-source-note', 'Every model retains its full aggregate, including failed trials, recovery retention, and evaluation provenance.'));
  for (const model of models) {
    const details = rawDetails(model, `${modelLabel(model)} · full aggregate`);
    const links = element('div', 'result-links');
    for (const [field, label] of [['report_json', 'Complete trials · JSON'], ['report_csv', 'Measurements · CSV'], ['trials_csv', 'Trial outcomes · CSV'], ['plot', 'Recorded curves']]) {
      const href = reportLink(model[field]);
      if (!href) continue;
      const link = element('a', '', label);
      link.href = href;
      if (field !== 'plot') link.download = '';
      links.append(link);
    }
    if (links.childElementCount) details.insertBefore(links, details.querySelector('pre'));
    sources.append(details);
  }
  const { models: reportedModels, results: reportedResults, ...context } = report;
  sources.append(rawDetails(context, 'Evaluation protocol, environment & summary provenance', 'result-card result-context'));
  holder.append(sources);
}
