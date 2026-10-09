(() => {
  'use strict';
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const config = window.MAPFLY_CONFIG || {};
  const scenes = [
    ['bigcity','Big City'],['nyc1950','NYC 1950'],['smallcity','Small City'],['brushifyurban','Brushify Urban'],['moderncity','Modern City'],
    ['realcitysf','Real City SF'],['citydowntown','City Downtown'],['abandonedcity','Abandoned City'],['battlefielddesert','Battlefield Desert'],['industrialcity','Industrial City'],
    ['laketown','Lake Town'],['moderncity2','Modern City 2'],['nordicharbour','Nordic Harbour'],['urbancity','Urban City'],['industrialarea','Industrial Area']
  ];
  const matchedScenes = scenes.filter(scene => scene[0] !== 'bigcity');
  const figures = {
    'p0r0': {title:'A fixed map.<br>A changing view.',description:'Start and goal markers stay fixed on the map. The policy receives a new first-person image at each step, without a live map-space position marker or an expert route.',position:'Static start',route:'None',map:'Unchanged',success:false,caption:'Dashed circles indicate live position for visualization only. The policy receives a fixed map.'},
    'p0r1': {title:'The full route.<br>A fixed starting point.',description:'The map provides fixed start and goal markers plus the complete expert route. This map input stays the same throughout the episode, while FPV continues to change.',position:'Static start',route:'Full expert route',map:'Unchanged',success:true,caption:'Dashed circles indicate live position for visualization only. Start, goal, and the full route are fixed policy inputs.'},
    'p1r0': {title:'A live position.<br>No prescribed route.',description:'The blue marker follows the UAV’s live map-space position at each decision step. The geographic background and goal remain fixed, and no expert route is shown.',position:'Live position',route:'None',map:'Position updates',success:true,caption:'The live map-space position is supplied from simulator ground truth. FPV updates at each step.'},
    'p1r1': {title:'Live position.<br>The remaining route.',description:'The map updates the UAV’s position and the remaining expert route at every step. The goal and geographic background remain fixed, with new FPV images for local navigation.',position:'Live position',route:'Remaining route',map:'Position + route update',success:true,caption:'The live marker and remaining route update using simulator ground-truth position.'}
  };
  const resultData = { unseen:[7.1,37.0,62.7,49.9], seen:[34.6,59.1,86.6,89.0] };
  const trackNames = ['P0–R0','P0–R1','P1–R0','P1–R1'];
  const policies = [
    ['Seq2Seq',[0.9,1.2,166.7,0.8,3.9,78.1],[1.8,1.9,162.5,1.7,5.4,65.9]],
    ['CMA',[1.4,1.5,166.8,1.1,4.5,86.2],[1.3,1.3,168.6,1.1,5.9,78.3]],
    ['ViNT',[45.6,51.6,50.8,40.6,46.6,34.0],[17.2,23.1,72.7,15.7,24.7,56.4]],
    ['UAV-Flow',[53.8,56.4,44.2,49.7,57.0,13.2],[28.2,30.8,64.3,26.6,35.9,32.0]],
    ['π₀',[52.9,58.4,57.8,47.0,51.1,17.0],[28.2,31.1,68.7,26.3,34.7,35.9]],
    ['MapFly-Agent (GR00T)',[59.6,65.0,33.8,52.8,54.6,17.0],[35.7,42.9,45.4,32.9,41.9,38.2]],
    ['MapFly-Agent (OFT)',[86.6,87.9,17.4,78.4,78.7,6.1],[62.7,65.3,36.9,58.2,55.6,26.5]]
  ];

  function html() {
    return [
      '<section class="section" id="abstract"><div class="container is-max-desktop"><div class="columns is-centered"><div class="column is-four-fifths"><h2 class="title is-3 has-text-centered">Abstract</h2><div class="content has-text-justified"><p>Navigation intent for unmanned aerial vehicles (UAVs) is commonly specified through route instructions or target descriptions. However, users may know a destination’s map location without knowing the route or being able to describe its surroundings. Thus, we introduce <strong>MapFly</strong>, a benchmark for <strong>prior-map-guided aerial visual navigation</strong>, where UAVs navigate using first-person view (FPV) and annotated prior maps that provide global spatial context and specify navigation intent.</p><p>MapFly defines four evaluation tracks by crossing static/live map-space position cues with route-free/route-assisted guidance, enabling controlled analysis of position cues and route guidance. It also provides an extensible toolkit for task creation, data collection, map rendering, and closed-loop evaluation. Using this toolkit, we construct <strong>MapFly-13K</strong>, a dataset of 13,300 validated episodes across 15 simulated outdoor scenes with matched map variants. We further develop <strong>MapFly-Agent</strong> as a reference policy for benchmark evaluation and input ablations. Experiments show substantial benefits from live position cues, but limited effective use of geographic map content. Meanwhile, navigation without live position cues remains challenging. We will publicly release the toolkit, dataset, and reference policy code upon publication.</p></div></div></div></div></section>',
      '<section class="section section-tint" id="overview"><div class="container is-max-desktop"><div class="section-heading has-text-centered"><h2 class="title is-3">Overview</h2></div><button class="figure-button" data-figure="assets/project-overview.webp" data-title="MapFly overview"><img src="assets/project-overview.webp" alt="MapFly ecosystem: extensible toolkit, MapFly-13K dataset, MapFly-Agent and four evaluation tracks." width="1830" height="618" loading="lazy"><span>Click to enlarge</span></button><div class="columns is-vcentered is-variable is-6 intent-row"><div class="column is-5 compact-copy"><h3 class="title is-4">Specifying navigation intent</h3><p>A prior map specifies the start and goal directly, with optional route guidance. The agent combines this global context with changing first-person observations during flight.</p></div><div class="column is-7"><button class="figure-button" data-figure="assets/intent-comparison.webp" data-title="Language-based and map-based intent"><img src="assets/intent-comparison.webp" alt="Comparison of detailed language instructions and prior-map-based intent specification." width="1118" height="428" loading="lazy"><span>Click to enlarge</span></button></div></div></div></section>',
      '<section class="section" id="tracks"><div class="container is-max-desktop"><div class="section-heading has-text-centered"><h2 class="title is-3">Four Evaluation Tracks</h2><p>Cross static or live position cues with route-free or route-assisted guidance. FPV updates at every step; the action interface is shared across all four tracks.</p></div><div class="track-map-grid">',
      [['p0r0','P0–R0','Fixed start and goal'],['p0r1','P0–R1','Fixed start, goal, and full route'],['p1r0','P1–R0','Live position, no route'],['p1r1','P1–R1','Live position and remaining route']].map(function(item){return '<button class="map-variant" data-figure="assets/track-map-'+item[0]+'.webp" data-title="'+item[1]+' map input"><img src="assets/track-map-'+item[0]+'.webp" alt="'+item[2]+'" width="224" height="224" loading="lazy"><strong>'+item[1]+'</strong><span>'+item[2]+'</span></button>';}).join(''),
      '</div><p class="figure-note">Blue: start or live position · Red: goal · Green: expert route. Live-position examples show an intermediate step. The geographic background and goal remain fixed.</p><div class="track-tabs" role="tablist" aria-label="Evaluation tracks">',
      [['p0r0','P0–R0','Static · Route-free'],['p0r1','P0–R1','Static · Route-assisted'],['p1r0','P1–R0','Live · Route-free'],['p1r1','P1–R1','Live · Route-assisted']].map(function(item,i){return '<button id="tab-'+item[0]+'" role="tab" aria-controls="track-panel" aria-selected="'+(i===0)+'" tabindex="'+(i===0?'0':'-1')+'" data-track="'+item[0]+'">'+item[1]+'<span>'+item[2]+'</span></button>';}).join(''),
      '</div><div class="columns is-variable is-5 track-demo" id="track-panel" role="tabpanel" aria-labelledby="tab-p0r0"><div class="column is-4"><p class="section-kicker">POLICY INPUT</p><h3 class="title" id="track-title">A fixed map.<br>A changing view.</h3><p id="track-description">Start and goal markers stay fixed on the map. The policy receives a new first-person image at each step, without a live map-space position marker or an expert route.</p><dl class="input-summary"><div><dt>Position cue</dt><dd id="position-value">Static start</dd></div><div><dt>Route guidance</dt><dd id="route-value">None</dd></div><div><dt>Map over time</dt><dd id="map-value">Unchanged</dd></div><div><dt>First-person view</dt><dd>Updates every step</dd></div></dl><p class="policy-flow">Map + FPV + state + instruction<br>↓<br><b>MapFly-Agent</b><br>↓<br>Waypoint increments + stop</p></div><div class="column is-8"><div class="demo-frame"><video id="track-video" controls muted playsinline preload="none" poster="assets/p0r0-poster.jpg" aria-label="P0–R0 navigation demonstration"><source src="assets/p0r0.mp4" type="video/mp4"></video><span class="speed-chip">10×</span><span class="outcome-chip is-failure" id="track-outcome">Failure</span></div><p class="demo-caption" id="track-caption">Dashed circles indicate live position for visualization only. The policy receives a fixed map.</p><p class="figure-note">Illustrative rollouts, not aggregate success rates. All tracks also receive start-relative state and a fixed task instruction.</p></div></div></div></section>',
      '<section class="section" id="toolkit"><div class="container is-max-desktop">',
      '<div class="section-heading has-text-centered"><p class="section-kicker">EXTENSIBLE TOOLKIT</p><h2 class="title is-2">From scene geometry to closed-loop evaluation.</h2><p>Built on Unreal Engine and AirSim, MapFly supports custom tasks, trajectory collection, map configurations, and policy evaluation through a common observation–action interface.</p></div>',
      '<div class="toolkit-steps"><article><span>01</span><h3>Construct tasks</h3><p>Define scenes, start–goal pairs, and episode parameters.</p></article><article><span>02</span><h3>Collect trajectories</h3><p>Plan, validate flights, and record synchronized observations.</p></article><article><span>03</span><h3>Render maps</h3><p>Produce matched map styles and task annotations.</p></article><article><span>04</span><h3>Evaluate policies</h3><p>Run closed-loop rollouts under a shared protocol.</p></article></div>',
      '<button class="figure-button paper-wide" data-figure="assets/toolkit.webp" data-title="MapFly toolkit and data generation"><img src="assets/toolkit.webp" alt="MapFly task construction, data collection, configurable map rendering and policy evaluation pipeline." width="2048" height="1108" loading="lazy"><span>Click to enlarge</span></button>',
      '<p class="figure-note">Trajectory generation → flight validation and synchronized recording → matched map rendering. Planning geometry is used offline only.</p></div></section>',

      '<section class="section section-tint" id="dataset"><div class="container is-max-desktop">',
      '<div class="section-heading has-text-centered"><p class="section-kicker">MAPFLY-13K</p><h2 class="title is-2">Matched maps. Diverse outdoor worlds.</h2><p>13,300 validated episodes across 15 simulated outdoor scenes. Every episode contains synchronized FPV, an expert trajectory, and matched map variants.</p></div>',
      '<div class="columns is-vcentered is-variable is-7"><div class="column is-7"><button class="figure-button" data-figure="assets/dataset-stats.webp" data-title="MapFly-13K dataset statistics"><img src="assets/dataset-stats.webp" alt="Dataset statistics covering trajectory distance, detour ratio, episodes by scene, and Train/Test splits." width="1118" height="742" loading="lazy"><span>Click to enlarge</span></button></div><div class="column is-5 content compact-copy"><h3 class="title is-3">Designed to test generalization.</h3><p>Train and Test Seen share 12 scenes; Test Unseen contains the remaining 3 held-out scenes. Every map variant for an episode remains in the same split.</p><div class="split-legend"><p><b class="swatch train"></b>Train <strong>9,660 · 72.6%</strong></p><p><b class="swatch seen"></b>Test Seen <strong>2,140 · 16.1%</strong></p><p><b class="swatch unseen"></b>Test Unseen <strong>1,500 · 11.3%</strong></p></div></div></div>',
      '<div class="gallery-heading"><div><p class="section-kicker">SCENE GALLERY</p><h3 class="title is-3">All 15 outdoor environments.</h3></div><p>Click a scene to inspect it at full resolution.</p></div><div class="scene-gallery" id="scene-gallery" aria-label="MapFly scene gallery"></div>',
      '<div class="matched-section"><div class="gallery-heading"><div><p class="section-kicker">MATCHED OBSERVATIONS</p><h3 class="title is-3">One episode. Multiple map views.</h3></div><label class="scene-picker" for="map-scene">Example scene<select id="map-scene" aria-label="Choose example scene"></select></label></div><p class="matched-copy">Compare first-person observations with OSM-style and satellite-style maps, using start–goal markers or route overlays. All variants share the same episode extent and annotations.</p><div class="matched-gallery" id="matched-gallery"></div><p class="figure-note" id="matched-scene-label" aria-live="polite"></p></div>',
      '</div></section>',

      '<section class="section" id="model"><div class="container is-max-desktop"><div class="section-heading has-text-centered"><p class="section-kicker">MAPFLY-AGENT</p><h2 class="title is-2">From visual context to waypoint actions.</h2><p>A reference policy adapted from StarVLA’s QwenOFT implementation combines Qwen3-VL-4B-Instruct with waypoint, stop, and auxiliary progress heads.</p></div><button class="figure-button model-figure" data-figure="assets/architecture.webp" data-title="MapFly-Agent architecture"><img src="assets/architecture.webp" alt="MapFly-Agent model architecture with FPV, prior map, fixed instruction, quantized state, action queries, and progress, stop, and action heads." width="986" height="594" loading="lazy"><span>Click to enlarge</span></button><p class="figure-note">The policy predicts eight waypoint increments and a stopping probability in one pass. The progress head is auxiliary supervision during training and is not used at inference.</p></div></section>',

      '<section class="section section-tint" id="results"><div class="container is-max-desktop"><div class="section-heading has-text-centered"><p class="section-kicker">EXPERIMENTAL FINDINGS</p><h2 class="title is-2">What helps an agent find its way?</h2><p>Controlled comparisons isolate the contributions of map-space position, route guidance, geographic map content, and first-person vision.</p></div>',
      '<div class="columns is-variable is-8"><div class="column is-7 result-card"><div class="result-card-head"><div><h3 class="title is-4">Success across the four tracks</h3><p>MapFly-Agent (OFT) · OSM-style maps + FPV</p></div><div class="result-switch" role="group" aria-label="Evaluation split"><button data-split="unseen" aria-pressed="true">Test Unseen</button><button data-split="seen" aria-pressed="false">Test Seen</button></div></div><div id="result-bars" aria-live="polite"></div><p class="figure-note">Each track is trained separately. Values from Table II.</p></div>',
      '<div class="column is-5 findings"><article><b>01</b><div><h3>Live position is a strong cue.</h3><p>On unseen scenes, route-free success rises from <strong>7.1% to 62.7%</strong> when a live map-space position is provided.</p></div></article><article><b>02</b><div><h3>Map content remains underused.</h3><p>On P1–R0, removing the geographic background raises unseen success from <strong>62.7% to 71.7%</strong>.</p></div></article><article><b>03</b><div><h3>Local vision matters.</h3><p>Without FPV, P1–R0 unseen success falls to <strong>33.3%</strong> and collisions rise from <strong>26.5% to 53.7%</strong>.</p></div></article></div></div>',
      '<div class="rollout-block"><div class="gallery-heading"><div><p class="section-kicker">QUALITATIVE ROLLOUTS</p><h3 class="title is-3">How the four policies navigate.</h3></div><p>Same start–goal layout, different position cues and route guidance.</p></div><button class="figure-button" data-figure="assets/qualitative-rollouts.webp" data-title="Four-track qualitative rollouts"><img src="assets/qualitative-rollouts.webp" alt="Four MapFly-Agent rollout examples across the benchmark tracks." width="2295" height="812" loading="lazy"><span>Click to enlarge</span></button><p class="figure-note">Figure 6. Dashed paths show expert trajectories; blue paths show executed trajectories. Route-free P0–R0 diverges, while P1–R0 approaches the goal.</p></div>',
      '<details class="paper-tables"><summary>View full experimental tables from the paper <span>＋</span></summary><div class="table-figures"><button class="figure-button" data-figure="assets/four-track-table.webp" data-title="Four-track evaluation — Table II"><img src="assets/four-track-table.webp" alt="Four-track evaluation results for Test Seen and Test Unseen." width="2295" height="529" loading="lazy"><span>Click to enlarge</span></button><button class="figure-button" data-figure="assets/input-ablation-table.webp" data-title="Input ablations — Table III"><img src="assets/input-ablation-table.webp" alt="Input ablation results for route-free tracks." width="2299" height="898" loading="lazy"><span>Click to enlarge</span></button></div></details>',
      '</div></section>',

      '<section class="section" id="video"><div class="container is-max-desktop"><div class="section-heading has-text-centered"><p class="section-kicker">PROJECT VIDEO</p><h2 class="title is-2">MapFly in 2 minutes 51 seconds.</h2><p>Overview of the toolkit, tracks, data, model, and evaluation examples.</p></div><div class="project-video"><video controls playsinline preload="metadata" poster="assets/hero-poster.jpg"><source src="assets/overview.mp4" type="video/mp4"></video></div></div></section>',

      '<section class="section section-tint" id="resources"><div class="container is-max-desktop"><div class="columns is-variable is-8 is-vcentered"><div class="column is-6"><p class="section-kicker">RESOURCES</p><h2 class="title is-2">Build on MapFly.</h2><p class="resource-copy">The toolkit, MapFly-13K dataset, and reference policy code are planned for public release upon publication.</p><div class="buttons"><a class="button is-dark is-rounded" href="assets/MapFly.pdf" target="_blank" rel="noopener">Read the paper</a><a class="button is-rounded" href="assets/overview.mp4" download>Download project video</a></div></div><div class="column is-6"><div class="release-list"><div><span>MapFly Toolkit</span><b data-resource-status="code">Coming soon</b></div><div><span>MapFly-13K Dataset</span><b data-resource-status="dataset">Coming soon</b></div><div><span>MapFly-Agent</span><b data-resource-status="code">Coming soon</b></div></div></div></div></div></section>',

      '<section class="section citation-section"><div class="container is-max-desktop"><div class="citation-card"><div class="citation-head"><div><p class="section-kicker">CITATION</p><h2 class="title is-4">Manuscript citation</h2></div><button class="button is-small is-rounded" id="copy-bibtex">Copy BibTeX</button></div><pre id="bibtex-code">@unpublished{mapfly,\n  title = {MapFly: A Benchmark for Prior-Map-Guided Aerial Visual Navigation},\n  author = {Anonymous Authors},\n  note = {Manuscript under review},\n  year = {2026}\n}</pre><p class="copy-feedback" id="copy-feedback" aria-live="polite"></p></div></div></section>'
    ].join('');
  }

  function makeGallery() {
    const gallery = $('#scene-gallery');
    gallery.innerHTML = scenes.map(function(scene) {
      const key = scene[0], name = scene[1], src = 'assets/scene-' + key + '.webp';
      return '<button class="scene-item" data-figure="' + src + '" data-title="' + name + '"><img src="' + src + '" alt="Example view of the ' + name + ' simulated scene." loading="lazy"><span>' + name + '</span></button>';
    }).join('');
    const select = $('#map-scene');
    if (!select) return;
    select.innerHTML = matchedScenes.map(function(scene) { return '<option value="' + scene[0] + '">' + scene[1] + '</option>'; }).join('');
    select.value = 'urbancity';
    select.addEventListener('change', renderMatchedGallery);
    renderMatchedGallery();
  }

  function renderMatchedGallery() {
    const select = $('#map-scene'), key = select.value, name = select.selectedOptions[0].textContent;
    const cards = [['obs','First-person view'],['osm_start_goal','OSM · Start & goal'],['osm_route','OSM · With route'],['satellite_start_goal','Satellite · Start & goal'],['satellite_route','Satellite · With route']];
    $('#matched-gallery').innerHTML = cards.map(function(card) {
      const src = 'assets/' + key + '-' + card[0] + '.webp';
      return '<button class="match-item" data-figure="' + src + '" data-title="' + name + ' — ' + card[1] + '"><img src="' + src + '" alt="' + card[1] + ' from a matched ' + name + ' episode." loading="lazy"><span>' + card[1] + '</span></button>';
    }).join('');
    $('#matched-scene-label').textContent = name + ' · Matched episode example';
  }

  function activateTrack(key) {
    const item = figures[key], video = $('#track-video'), outcome = $('#track-outcome');
    video.pause();
    video.poster = 'assets/' + key + '-poster.jpg';
    video.src = 'assets/' + key + '.mp4';
    video.load();
    video.setAttribute('aria-label', key.toUpperCase() + ' navigation demonstration');
    $('#track-panel').setAttribute('aria-labelledby', 'tab-' + key);
    $('#track-title').innerHTML = item.title;
    $('#track-description').textContent = item.description;
    $('#position-value').textContent = item.position;
    $('#route-value').textContent = item.route;
    $('#map-value').textContent = item.map;
    $('#track-caption').textContent = item.caption;
    outcome.textContent = item.success ? 'Success' : 'Failure';
    outcome.classList.toggle('is-failure', !item.success);
    $$('[data-track]').forEach(function(button) {
      button.setAttribute('aria-selected', String(button.dataset.track === key));
      button.tabIndex = button.dataset.track === key ? 0 : -1;
    });
  }

  function updateResults(split) {
    if (!$('#result-bars')) return;
    const values = resultData[split], best = Math.max.apply(null, values);
    $('#result-bars').innerHTML = values.map(function(value, i) {
      return '<div class="bar-row ' + (value === best ? 'best' : '') + '" aria-label="' + trackNames[i] + ': ' + value.toFixed(1) + ' percent success"><span class="bar-label">' + trackNames[i] + '</span><div class="bar-track"><div class="bar-fill" style="width:' + value + '%"></div></div><span class="bar-value">' + value.toFixed(1) + '</span></div>';
    }).join('');
    $$('[data-split]').forEach(function(button) { button.setAttribute('aria-pressed', String(button.dataset.split === split)); });
    $('#policy-table').innerHTML=policies.map(function(row,i){return '<tr'+(i===6?' class="highlight"':'')+'><th scope="row">'+row[0]+'</th>'+row[split==='seen'?1:2].map(function(v){return '<td>'+v.toFixed(1)+'</td>';}).join('')+'</tr>';}).join('');
    $('#table-split-label').textContent=split==='seen'?'Test Seen':'Test Unseen';
  }

  function openFigure(button) {
    const dialog = $('#figure-dialog'), image = $('#enlarged-figure');
    image.src = button.dataset.figure;
    image.alt = button.querySelector('img') ? button.querySelector('img').alt : button.dataset.title;
    $('#figure-title').textContent = button.dataset.title || 'Figure';
    dialog.showModal();
  }

  function setResources() {
    if (config.authors) $('#authors').textContent = config.authors;
    if (config.publication) $('#publication').textContent = config.publication;
    if (config.affiliation) { const affiliation=document.createElement('p'); affiliation.className='affiliation'; affiliation.textContent=config.affiliation; $('#publication').after(affiliation); }
    if (config.bibtex) $('#bibtex-code').textContent = config.bibtex;
    ['code','dataset'].forEach(function(type) {
      const url = config[type + 'Url'];
      if (!url || !/^https?:\/\//i.test(url)) return;
      const link = $('#' + type + '-link');
      link.href = url; link.target = '_blank'; link.rel = 'noopener';
      link.classList.remove('resource-pending');
      link.removeAttribute('aria-disabled');
      const small = $('small', link); if (small) small.textContent = 'Available';
      $$('[data-resource-status="' + type + '"]').forEach(function(status) { status.textContent = 'Available'; });
    });
  }

  function copyBibtex() {
    const value = $('#bibtex-code').textContent, feedback = $('#copy-feedback'), button = $('#copy-bibtex');
    const success = function() { feedback.textContent = 'BibTeX copied to clipboard.'; button.textContent = 'Copied'; setTimeout(function(){button.textContent='Copy BibTeX';},1800); };
    if (navigator.clipboard) navigator.clipboard.writeText(value).then(success).catch(fallback);
    else fallback();
    function fallback() {
      const input = document.createElement('textarea'); input.value = value; input.style.position='fixed'; input.style.opacity='0';
      document.body.appendChild(input); input.select(); document.execCommand('copy'); input.remove(); success();
    }
  }

  function arrangePage() {
    const content=$('#mapfly-content');
    // Cover → Abstract → Video → Scenes → Toolchain → Dataset → Model → Examples → Results → BibTeX.
    const overview=$('#overview');
    const cover=$('.figure-button',overview);
    $('img',cover).loading='eager';
    $('img',cover).setAttribute('fetchpriority','high');
    $('#cover-content').append(cover);
    $('#abstract .container').append($('.intent-row',overview));
    overview.remove();

    const scenesSection=document.createElement('section');
    scenesSection.className='section section-tint';scenesSection.id='scenes';
    scenesSection.innerHTML='<div class="container is-max-desktop"><div class="section-heading has-text-centered"><h2 class="title is-2">Simulation Environments</h2><p>Fifteen outdoor worlds spanning urban neighborhoods, industrial districts, and arid landscapes.</p></div></div>';
    const scenesContainer=$('.container',scenesSection);
    scenesContainer.append($('.scene-mosaic'),$('#scene-gallery'));
    $('#dataset .gallery-heading').remove();
    content.append(scenesSection);

    $('#dataset .section-heading').after($('.benchmark-stats'));
    $('#benchmark-stats-source').remove();
    $('#tracks .container').append($('#results .rollout-block'));
    const citation=$('.citation-section');citation.id='bibtex';
    const resources=$('#resources');
    const resourceDetails=document.createElement('details');resourceDetails.className='resource-details';resourceDetails.id='resources';
    resourceDetails.innerHTML='<summary>Code &amp; dataset release</summary>';
    resourceDetails.append($('.columns',resources));
    $('#dataset .container').append(resourceDetails);resources.remove();

    const headings={abstract:'Abstract',video:'Project Video',scenes:'Simulation Environments',toolkit:'Data Generation Toolchain',dataset:'Dataset Overview',model:'MapFly-Agent',tracks:'Navigation Examples',results:'Experimental Results',bibtex:'BibTeX'};
    Object.entries(headings).forEach(function(entry){const section=$('#'+entry[0]);const title=$('h2',section);title.textContent=entry[1];title.className='title is-2';$$('.section-heading > .section-kicker',section).forEach(function(k){k.remove();});content.append(section);});
    $('#abstract h2').classList.add('has-text-centered');
    $('#tracks .section-heading > p').textContent='Explore four evaluation tracks defined by static or live position cues and route-free or route-assisted guidance. Each track uses the same waypoint-action interface.';
    $('#video .section-heading > p').textContent='A 2 min 51 sec introduction to MapFly: the toolkit, benchmark, dataset, policy, and flight demonstrations.';
    $('#dataset .section-heading > p').textContent='MapFly-13K contains 13,300 validated episodes, with synchronized first-person observations, expert trajectories, and matched prior-map variants.';
    $('#toolkit .section-heading > p').textContent='From scene geometry to validated flights and matched maps. The extensible toolkit supports task construction, data collection, map rendering, and closed-loop policy evaluation.';
    $('#resources h2').textContent='Resources';
  }

  function simplifyPresentation() {
    // Paper-style presentation: prose, visual evidence, then a short explanation.
    $$('.section-kicker').forEach(function(el){el.remove();});
    $('.toolkit-steps').remove();
    $('#toolkit .figure-note').textContent='Expert trajectories are planned from scene geometry, validated through flight in AirSim, and retained with synchronized observations. Matched maps are then rendered from the validated episodes. The geometry used for planning is available offline only.';

    const dataset=$('#dataset .container');
    $('.benchmark-stats',dataset).remove();
    $('.columns',dataset).remove();
    $('#dataset .section-heading > p').textContent='MapFly-13K contains 13,300 validated navigation episodes across 15 simulated outdoor scenes. Each episode includes a start–goal task, an expert trajectory, synchronized first-person images, and matched prior maps. Expert trajectories have a mean length of 223.8 m. The plots below summarize navigation distance, detour ratio, episode counts by scene, and dataset splits.';
    const charts=[
      ['traj_length_hist.png','Navigation distance','Distribution of expert trajectory lengths in meters.'],
      ['detour_ratio_hist.png','Detour ratio','Expert trajectory length divided by straight-line start–goal distance.'],
      ['scene_distribution.png','Episodes by scene','Number of validated episodes in each of the 15 scenes.'],
      ['split_seen12.png','Dataset splits','Train: 72.6%; Test Seen: 16.1%; Test Unseen: 11.3%.']
    ];
    $('.section-heading',dataset).insertAdjacentHTML('afterend','<div class="dataset-plot-grid">'+charts.map(function(c,i){return '<figure><button class="plain-plot" data-figure="assets/'+c[0]+'" data-title="'+c[1]+'"><img src="assets/'+c[0]+'" alt="'+c[2]+'" loading="lazy"></button><figcaption>('+String.fromCharCode(97+i)+') '+c[1]+'</figcaption></figure>';}).join('')+'</div><p class="paper-paragraph dataset-split-note">Training and Test Seen contain 9,660 and 2,140 episodes from the same 12 scenes. Test Unseen contains 1,500 episodes from three held-out scenes. All map variants of a given episode remain in the same split, enabling comparisons under matched navigation tasks.</p>');
    $('.matched-section h3',dataset).textContent='Matched observations and prior maps';
    $('.matched-copy',dataset).textContent='The same episode can be represented with an OSM-style or satellite-style map, with start–goal markers and an optional route. The examples below show these matched inputs alongside a first-person observation.';
    const resources=$('#resources');
    resources.innerHTML='<summary>Code and dataset availability</summary><p class="paper-paragraph">The toolkit, MapFly-13K dataset, and reference policy code are planned for public release upon publication. <a href="assets/MapFly.pdf" target="_blank" rel="noopener">Read the paper</a> or <a href="assets/overview.mp4" download>download the project video</a>.</p><p class="release-inline">Code: <span data-resource-status="code">Coming soon</span> · Dataset: <span data-resource-status="dataset">Coming soon</span></p>';

    const examples=$('#tracks .container');
    $('.rollout-block',examples).remove();
    $('.track-tabs',examples).remove();
    $('.track-demo',examples).remove();
    $('.track-map-grid',examples).remove();
    $('.figure-note',examples).remove();
    $('#tracks .section-heading > p').textContent='The four tracks combine static or live map-space position cues with route-free or route-assisted guidance. Static tracks receive the same map at every step; live tracks update the position marker and, when present, the remaining route. First-person images change at every step. The demonstrations below use the same waypoint-action interface and are shown at 10× speed.';
    const demos=[['p0r0','P0–R0 · Static position, route-free'],['p0r1','P0–R1 · Static position, route-assisted'],['p1r0','P1–R0 · Live position, route-free'],['p1r1','P1–R1 · Live position, route-assisted']];
    examples.insertAdjacentHTML('beforeend','<div class="demo-gallery">'+demos.map(function(d){const item=figures[d[0]];return '<figure class="demo-example"><h3>'+d[1]+'</h3><div class="demo-frame"><video controls muted playsinline preload="none" poster="assets/'+d[0]+'-poster.jpg" aria-label="'+d[1]+' demonstration"><source src="assets/'+d[0]+'.mp4" type="video/mp4"></video><span class="speed-chip">10×</span><span class="outcome-chip '+(item.success?'':'is-failure')+'">'+(item.success?'Success':'Failure')+'</span></div><figcaption>'+item.caption+'</figcaption></figure>';}).join('')+'</div><p class="paper-paragraph examples-note">Dashed position circles in the two static-map demonstrations are visualization aids only; they are not part of the policy input. These are illustrative rollouts rather than aggregate success rates. All tracks also receive start-relative state and a fixed task instruction.</p>');

    const resultColumns=$('#results .columns');
    $('.findings',resultColumns).remove();
    resultColumns.className='results-chart-wrap';
    $('.result-card',resultColumns).className='result-card';
    resultColumns.insertAdjacentHTML('afterend','<p class="paper-paragraph result-discussion">On unseen scenes, adding a live position marker improves route-free success from 7.1% to 62.7%. Under live grounding (P1–R0), removing the geographic background increases success to 71.7%, suggesting that the evaluated policy does not yet make effective use of map structure. Removing first-person vision reduces success to 33.3% and raises the collision rate from 26.5% to 53.7%, demonstrating the importance of local visual observations.</p>');
    $('.gallery-heading > p',$('#scenes'))?.remove();
    $('#abstract .intent-row').remove();
    $('#dataset .matched-section').remove();
    $('#results').remove();
    $('.section-nav a[href="#results"]').remove();
    $('#resources').remove();
    $('#scenes .scene-mosaic').remove();
    $('#bibtex').insertAdjacentHTML('beforebegin', '<section class="section paper-section" id="paper"><div class="container is-max-desktop"><div class="section-heading has-text-centered"><h2 class="title is-2">Paper</h2></div><iframe class="paper-reader" src="assets/MapFly.pdf" title="MapFly full paper — PDF reader" width="100%" height="550"></iframe><div class="paper-reader-links"><a href="assets/MapFly.pdf" target="_blank" rel="noopener">Open PDF in a new tab ↗</a><a href="assets/MapFly.pdf" download="MapFly_final.pdf">Download PDF ↓</a></div></div></section>');
    $('.section-nav a[href="#bibtex"]').insertAdjacentHTML('beforebegin', '<a href="#paper">Paper</a>');
    // Keep the map visible: report the outcome underneath each video.
    $$('.demo-example').forEach(function(figure){
      const outcome=$('.outcome-chip',figure);
      const line=document.createElement('div');line.className='demo-outcome';
      line.append(outcome);
      $('.demo-frame',figure).after(line);
      const heading=$('h3',figure);
      heading.textContent=heading.textContent.replace('Static position,','Static ·').replace('Live position,','Live ·');
    });
  }

  document.addEventListener('DOMContentLoaded', function() {
    $('#mapfly-content').innerHTML = html();
    $('#dataset .gallery-heading').insertAdjacentHTML('beforebegin','<figure class="scene-mosaic"><video controls muted loop playsinline preload="none" poster="assets/scene-mosaic-poster.jpg" aria-label="Nine example scenes from MapFly"><source src="assets/scene-mosaic.mp4" type="video/mp4"></video><figcaption class="figure-note has-text-centered">A selection of nine scenes from MapFly-13K.</figcaption></figure>');
    $('#results .rollout-block').insertAdjacentHTML('beforebegin','<details class="paper-tables"><summary>View policy comparison — Table I</summary><p class="figure-note">P1–R0 with OSM-style maps and FPV. SR, OSR, SPL, nDTW and CR use a 0–100 scale; NE is measured in meters.</p><div class="table-scroll" tabindex="0" aria-label="Policy results; scroll horizontally on small screens"><table class="policy-table"><caption>Policy comparison on <span id="table-split-label">Test Unseen</span></caption><thead><tr><th scope="col">Policy</th><th scope="col">SR ↑</th><th scope="col">OSR ↑</th><th scope="col">NE ↓</th><th scope="col">SPL ↑</th><th scope="col">nDTW ↑</th><th scope="col">CR ↓</th></tr></thead><tbody id="policy-table"></tbody></table></div></details>');
    $('#results .rollout-block .figure-note').textContent='Figure 6. Dashed paths show expert trajectories; blue paths show executed trajectories. Samples 1–3 are selected moments, not consecutive timesteps. Expert paths in the top row are shown for comparison; only R1 provides route overlays to the policy. These examples are separate from the video demonstrations above.';
    arrangePage();
    simplifyPresentation();
    setResources(); makeGallery(); updateResults('unseen');
    $$('[data-split]').forEach(function(button) { button.addEventListener('click', function(){ updateResults(button.dataset.split); }); });
    document.addEventListener('click', function(event) {
      const trigger = event.target.closest('[data-figure]');
      if (trigger) openFigure(trigger);
    });
    $('#close-figure').addEventListener('click', function(){ $('#figure-dialog').close(); });
    $('#figure-dialog').addEventListener('click', function(event) {
      const rect = event.currentTarget.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.currentTarget.close();
    });
    $('#copy-bibtex').addEventListener('click', copyBibtex);
    const scrollButton = $('#scroll-to-top');
    window.addEventListener('scroll', function(){ scrollButton.classList.toggle('visible', window.scrollY > 420); }, {passive:true});
    scrollButton.addEventListener('click', function(){ window.scrollTo({top:0,behavior:'smooth'}); });
    document.addEventListener('play',function(event){if(event.target.tagName==='VIDEO')$$('video').forEach(function(v){if(v!==event.target)v.pause();});},true);
    document.addEventListener('visibilitychange',function(){if(document.hidden)$$('video').forEach(function(v){v.pause();});});
  });
})();
