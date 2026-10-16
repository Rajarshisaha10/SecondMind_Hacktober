/* Second Mind Caregiver Dashboard - Client Application Logic */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  loadHealthStatus();
  loadProfile();
  loadSchedule();
  loadMemoryLog();
  loadRegisteredObjects();
  loadRecentMatches();

  // Attach Event Listeners
  document.getElementById('profileForm').addEventListener('submit', handleProfileSubmit);
  document.getElementById('scheduleForm').addEventListener('submit', handleScheduleSubmit);
  document.getElementById('addMedBtn').addEventListener('click', () => addMedicationRow());
  document.getElementById('addFamilyBtn').addEventListener('click', () => addFamilyRow());
  document.getElementById('addAppBtn').addEventListener('click', () => addAppointmentRow());
  document.getElementById('addRoutineBtn').addEventListener('click', () => addRoutineRow());
  document.getElementById('addCustomBtn').addEventListener('click', () => addCustomRow());
  document.getElementById('refreshMemBtn').addEventListener('click', loadMemoryLog);
  document.getElementById('refreshHealthBtn').addEventListener('click', loadHealthStatus);

  // Object Registry & Visual Search Listeners
  const regForm = document.getElementById('registerObjectForm');
  if (regForm) regForm.addEventListener('submit', handleObjectRegisterSubmit);
  const searchForm = document.getElementById('testVisualSearchForm');
  if (searchForm) searchForm.addEventListener('submit', handleVisualSearchSubmit);
  const refObjBtn = document.getElementById('refreshObjectsBtn');
  if (refObjBtn) refObjBtn.addEventListener('click', loadRegisteredObjects);
  const refMatchBtn = document.getElementById('refreshMatchesBtn');
  if (refMatchBtn) refMatchBtn.addEventListener('click', loadRecentMatches);
  const refRealityBtn = document.getElementById('refreshRealityBtn');
  if (refRealityBtn) refRealityBtn.addEventListener('click', loadRealityChecksum);

  // Initialize Voice Assistant
  initVoiceAssistant();
});

/* Tab Navigation */
function initTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.dataset.tab;
      document.getElementById(target).classList.add('active');

      if (target === 'memory-tab') loadMemoryLog();
      if (target === 'health-tab') loadHealthStatus();
      if (target === 'objects-tab') {
        loadRegisteredObjects();
        loadRecentMatches();
      }
      if (target === 'reality-tab') {
        loadRealityChecksum();
      }
    });
  });
}

/* Toast Notifications */
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerText = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.remove();
  }, 4000);
}

/* System Health Loading */
async function loadHealthStatus() {
  const badge = document.getElementById('systemStatusBadge');
  try {
    const res = await fetch('/health');
    const data = await res.json();
    if (data.status === 'healthy') {
      badge.className = 'status-badge';
      badge.innerHTML = `<span class="status-dot"></span> SYSTEM HEALTHY`;
    } else {
      badge.className = 'status-badge warning';
      badge.innerHTML = `DEGRADED (${data.ollama})`;
    }

    document.getElementById('healthOllamaVal').innerText = data.ollama;
    document.getElementById('healthDbVal').innerText = data.database;
    document.getElementById('healthAgentsVal').innerText = `${data.agents} Active Agents`;
    document.getElementById('healthUptimeVal').innerText = `${data.uptime_seconds}s`;
  } catch (e) {
    badge.className = 'status-badge error';
    badge.innerHTML = `SYSTEM OFFLINE`;
  }
}

/* Profile Tab Loading & Handling */
async function loadProfile() {
  try {
    const res = await fetch('/api/profile');
    const data = await res.json();
    const patient = data.patient || {};

    document.getElementById('patientName').value = patient.name || '';
    document.getElementById('patientAge').value = patient.age || '';
    document.getElementById('patientConditions').value = (patient.conditions || []).join(', ');

    // Populate Medications
    const medContainer = document.getElementById('medicationsContainer');
    medContainer.innerHTML = '';
    (patient.medications || []).forEach(med => addMedicationRow(med));

    // Populate Family Contacts
    const familyContainer = document.getElementById('familyContainer');
    familyContainer.innerHTML = '';
    (data.family || []).forEach(fam => addFamilyRow(fam));
  } catch (e) {
    showToast('Failed to load profile data', 'error');
  }
}

function addMedicationRow(data = {}) {
  const container = document.getElementById('medicationsContainer');
  const row = document.createElement('div');
  row.className = 'item-row med-row';
  row.innerHTML = `
    <div class="form-group">
      <input type="text" class="med-name" placeholder="Medication Name" value="${data.name || ''}" required>
    </div>
    <div class="form-group">
      <input type="text" class="med-times" placeholder="Times e.g. 08:00, 20:00" value="${(data.schedule_times || []).join(', ')}" required>
    </div>
    <div class="form-group">
      <input type="number" class="med-grace" placeholder="Grace Min" value="${data.grace_window_minutes || 30}" required>
    </div>
    <button type="button" class="btn-remove" onclick="this.parentElement.remove()">✕</button>
  `;
  container.appendChild(row);
}

function addFamilyRow(data = {}) {
  const container = document.getElementById('familyContainer');
  const row = document.createElement('div');
  row.className = 'item-row family-row';
  row.innerHTML = `
    <div class="form-group">
      <input type="text" class="fam-name" placeholder="Contact Name" value="${data.name || ''}" required>
    </div>
    <div class="form-group">
      <input type="text" class="fam-rel" placeholder="Relationship e.g. Daughter" value="${data.relationship || ''}" required>
    </div>
    <div class="form-group">
      <input type="text" class="fam-phone" placeholder="Phone e.g. 555-0199" value="${data.phone || ''}" required>
    </div>
    <button type="button" class="btn-remove" onclick="this.parentElement.remove()">✕</button>
  `;
  container.appendChild(row);
}

async function handleProfileSubmit(e) {
  e.preventDefault();
  const name = document.getElementById('patientName').value;
  const age = parseInt(document.getElementById('patientAge').value) || 0;
  const conditions = document.getElementById('patientConditions').value.split(',').map(s => s.trim()).filter(Boolean);

  const medications = Array.from(document.querySelectorAll('.med-row')).map(row => ({
    name: row.querySelector('.med-name').value.trim(),
    schedule_times: row.querySelector('.med-times').value.split(',').map(s => s.trim()).filter(Boolean),
    grace_window_minutes: parseInt(row.querySelector('.med-grace').value) || 30,
  }));

  const family = Array.from(document.querySelectorAll('.family-row')).map(row => ({
    name: row.querySelector('.fam-name').value.trim(),
    relationship: row.querySelector('.fam-rel').value.trim(),
    phone: row.querySelector('.fam-phone').value.trim(),
  }));

  const payload = {
    patient: { name, age, conditions, medications },
    family
  };

  try {
    const res = await fetch('/api/profile', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      showToast('Patient Profile saved successfully!');
    } else {
      showToast('Failed to save profile', 'error');
    }
  } catch (err) {
    showToast('Error saving profile', 'error');
  }
}

/* Schedule Tab Loading & Handling */
async function loadSchedule() {
  try {
    const res = await fetch('/api/schedule');
    const data = await res.json();

    const appContainer = document.getElementById('appointmentsContainer');
    appContainer.innerHTML = '';
    (data.appointments || []).forEach(app => addAppointmentRow(app));

    const rtContainer = document.getElementById('routinesContainer');
    rtContainer.innerHTML = '';
    (data.routines || []).forEach(rt => addRoutineRow(rt));

    const customContainer = document.getElementById('customRemindersContainer');
    customContainer.innerHTML = '';
    (data.custom_reminders || []).forEach(rem => addCustomRow(rem));
  } catch (e) {
    showToast('Failed to load schedule data', 'error');
  }
}

function addAppointmentRow(data = {}) {
  const container = document.getElementById('appointmentsContainer');
  const row = document.createElement('div');
  row.className = 'item-row app-row';
  row.innerHTML = `
    <div class="form-group">
      <input type="text" class="app-title" placeholder="Appointment Title" value="${data.title || ''}" required>
    </div>
    <div class="form-group">
      <input type="text" class="app-dt" placeholder="DateTime (e.g. 2026-07-31T10:00:00)" value="${data.datetime || ''}" required>
    </div>
    <div class="form-group">
      <input type="text" class="app-loc" placeholder="Location" value="${data.location || ''}">
    </div>
    <button type="button" class="btn-remove" onclick="this.parentElement.remove()">✕</button>
  `;
  container.appendChild(row);
}

function addRoutineRow(data = {}) {
  const container = document.getElementById('routinesContainer');
  const row = document.createElement('div');
  row.className = 'item-row rt-row';
  row.innerHTML = `
    <div class="form-group">
      <input type="text" class="rt-name" placeholder="Routine Name" value="${data.name || ''}" required>
    </div>
    <div class="form-group">
      <input type="text" class="rt-time" placeholder="Start Time e.g. 07:30" value="${data.start_time || ''}" required>
    </div>
    <div class="form-group">
      <input type="text" class="rt-steps" placeholder="Steps (comma separated)" value="${(data.steps || []).join(', ')}">
    </div>
    <button type="button" class="btn-remove" onclick="this.parentElement.remove()">✕</button>
  `;
  container.appendChild(row);
}

function addCustomRow(data = {}) {
  const container = document.getElementById('customRemindersContainer');
  const row = document.createElement('div');
  row.className = 'item-row custom-row';
  row.innerHTML = `
    <div class="form-group">
      <input type="text" class="custom-msg" placeholder="Reminder Message" value="${data.message || ''}" required>
    </div>
    <div class="form-group">
      <input type="text" class="custom-time" placeholder="Trigger Time e.g. 15:00" value="${data.trigger_time || ''}" required>
    </div>
    <button type="button" class="btn-remove" onclick="this.parentElement.remove()">✕</button>
  `;
  container.appendChild(row);
}

async function handleScheduleSubmit(e) {
  e.preventDefault();
  const appointments = Array.from(document.querySelectorAll('.app-row')).map((row, i) => ({
    id: `app-00${i+1}`,
    title: row.querySelector('.app-title').value.trim(),
    datetime: row.querySelector('.app-dt').value.trim(),
    location: row.querySelector('.app-loc').value.trim(),
    reminder_lead_minutes: [30, 10]
  }));

  const routines = Array.from(document.querySelectorAll('.rt-row')).map((row, i) => ({
    id: `rt-00${i+1}`,
    name: row.querySelector('.rt-name').value.trim(),
    days: ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    start_time: row.querySelector('.rt-time').value.trim(),
    steps: row.querySelector('.rt-steps').value.split(',').map(s => s.trim()).filter(Boolean)
  }));

  const custom_reminders = Array.from(document.querySelectorAll('.custom-row')).map((row, i) => ({
    id: `rem-00${i+1}`,
    message: row.querySelector('.custom-msg').value.trim(),
    trigger_time: row.querySelector('.custom-time').value.trim(),
    days: ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
    repeat: true
  }));

  const payload = { appointments, routines, custom_reminders };

  try {
    const res = await fetch('/api/schedule', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      showToast('Schedule Setup saved successfully!');
    } else {
      showToast('Failed to save schedule', 'error');
    }
  } catch (err) {
    showToast('Error saving schedule', 'error');
  }
}

/* Live Memory Log Loading */
async function loadMemoryLog() {
  const tbody = document.getElementById('memoryLogTbody');
  try {
    const res = await fetch('/api/observations');
    const observations = await res.json();

    if (!observations || observations.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="text-center">No room observations recorded yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = observations.map(obs => {
      let badgeClass = 'badge-object';
      if (obs.event_type === 'medication_taken') badgeClass = 'badge-med';
      if (obs.event_type === 'help_request') badgeClass = 'badge-help';

      return `
        <tr>
          <td><code title="${obs.observed_at || obs.timestamp || ''}">${obs.formatted_time || obs.timestamp || 'Time unavailable'}</code></td>
          <td><span class="badge-event ${badgeClass}">${obs.event_type}</span></td>
          <td><strong>${obs.object || ''}</strong></td>
          <td>${obs.location || ''}</td>
          <td>${obs.action || ''}</td>
          <td>${obs.raw_description || ''}</td>
        </tr>
      `;
    }).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-center">Failed to load room memory observations.</td></tr>`;
  }
}

/* Object Registry & Visual Search Functions */
async function loadRegisteredObjects() {
  const tbody = document.getElementById('objectsTableBody');
  const targetSelect = document.getElementById('testTargetObj');
  if (!tbody) return;

  try {
    const res = await fetch('/api/objects');
    const objects = await res.json();

    if (targetSelect) {
      targetSelect.innerHTML = '<option value="">All Registered Objects</option>' +
        objects.map(o => `<option value="${o.id}">${o.name}</option>`).join('');
    }

    if (!objects || objects.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="text-center">No objects registered yet. Register an object above to begin.</td></tr>`;
      return;
    }

    tbody.innerHTML = objects.map(obj => `
      <tr>
        <td><strong>${obj.name}</strong></td>
        <td><span class="badge-event badge-object">${obj.category || 'personal'}</span></td>
        <td>${obj.description || '—'}</td>
        <td><span class="badge-event" style="background:#000;color:#fff;">${obj.reference_count || 0} ref(s)</span></td>
        <td><code>${obj.created_at ? obj.created_at.split('T')[0] : ''}</code></td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="promptAddReference('${obj.id}', '${obj.name}')" style="margin-right: 6px;">+ Ref</button>
          <button class="btn btn-sm" onclick="deleteObject('${obj.id}')" style="background:#FF3333;color:#fff;">Delete</button>
        </td>
      </tr>
    `).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="6" class="text-center">Failed to load registered objects.</td></tr>`;
  }
}

async function handleObjectRegisterSubmit(e) {
  e.preventDefault();
  const btn = document.getElementById('saveObjBtn');
  btn.disabled = true;
  btn.innerText = 'Registering...';

  const name = document.getElementById('objName').value.trim();
  const category = document.getElementById('objCategory').value;
  const description = document.getElementById('objDescription').value.trim();
  const fileInput = document.getElementById('objImageFile');

  const formData = new FormData();
  formData.append('name', name);
  formData.append('category', category);
  formData.append('description', description);
  if (fileInput.files.length > 0) {
    formData.append('file', fileInput.files[0]);
  }

  try {
    const res = await fetch('/api/objects', {
      method: 'POST',
      body: formData,
    });
    if (res.ok) {
      showToast(`Registered '${name}' successfully!`);
      document.getElementById('registerObjectForm').reset();
      loadRegisteredObjects();
    } else {
      const err = await res.json();
      showToast(err.detail || 'Registration failed', 'error');
    }
  } catch (err) {
    showToast('Network error during registration', 'error');
  } finally {
    btn.disabled = false;
    btn.innerText = 'Register Object';
  }
}

async function deleteObject(objectId) {
  if (!confirm('Are you sure you want to delete this registered object and its reference embeddings?')) return;
  try {
    const res = await fetch(`/api/objects/${objectId}`, { method: 'DELETE' });
    if (res.ok) {
      showToast('Object deleted.');
      loadRegisteredObjects();
    } else {
      showToast('Failed to delete object', 'error');
    }
  } catch (e) {
    showToast('Error deleting object', 'error');
  }
}

async function promptAddReference(objectId, objectName) {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/jpeg,image/png,image/webp';
  input.onchange = async () => {
    if (input.files.length === 0) return;
    const formData = new FormData();
    formData.append('file', input.files[0]);
    showToast(`Uploading additional reference for ${objectName}...`);
    try {
      const res = await fetch(`/api/objects/${objectId}/references`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        showToast('Reference image added and embedded!');
        loadRegisteredObjects();
      } else {
        const err = await res.json();
        showToast(err.detail || 'Upload failed', 'error');
      }
    } catch (e) {
      showToast('Error uploading reference', 'error');
    }
  };
  input.click();
}

async function handleVisualSearchSubmit(e) {
  e.preventDefault();
  const btn = document.getElementById('runSearchBtn');
  const resultBox = document.getElementById('visualSearchResultBox');
  const fileInput = document.getElementById('testFrameFile');
  const targetSelect = document.getElementById('testTargetObj');

  if (fileInput.files.length === 0) {
    showToast('Select an image file to search.', 'warning');
    return;
  }

  btn.disabled = true;
  btn.innerText = 'Analyzing Frame...';

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  if (targetSelect.value) {
    formData.append('target_object_id', targetSelect.value);
  }
  formData.append('save_observation', 'true');

  try {
    const res = await fetch('/api/visual-search', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    resultBox.style.display = 'block';

    let statusColor = '#FFB800';
    if (data.status === 'strong_match') statusColor = '#00B865';
    if (data.status === 'no_match' || data.status === 'failed') statusColor = '#FF3333';

    if (data.best_match) {
      const bm = data.best_match;
      resultBox.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
          <span style="font-weight:800; text-transform:uppercase; padding:4px 8px; border:2px solid #000; background:${statusColor}; color:#fff; border-radius:4px;">
            ${data.status.replace('_', ' ')}
          </span>
          <span style="font-weight:600; font-size:12px;">Latency: ${data.latency_ms}ms</span>
        </div>
        <p style="font-size:15px; margin-bottom:4px;"><strong>Target:</strong> ${bm.object_name}</p>
        <p style="font-size:13px; margin-bottom:4px;"><strong>Approximate Location:</strong> ${bm.spatial_location}</p>
        <p style="font-size:13px; margin-bottom:4px;"><strong>Observed Time:</strong> ${data.formatted_time || data.frame_timestamp || 'Time unavailable'}</p>
        <p style="font-size:13px; margin-bottom:4px;"><strong>Cosine Similarity:</strong> ${(bm.similarity_score * 100).toFixed(1)}%</p>
        <p style="font-size:12px; color:var(--text-muted);">Bounding Box: [x=${bm.bounding_box.x}, y=${bm.bounding_box.y}, w=${bm.bounding_box.width}, h=${bm.bounding_box.height}]</p>
        ${bm.local_matches ? `<p style="font-size:12px; color:#00B865; font-weight:600;">✓ SIFT/ORB Local Keypoint Inliers: ${bm.local_matches}</p>` : ''}
      `;
    } else {
      resultBox.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <span style="font-weight:800; text-transform:uppercase; padding:4px 8px; border:2px solid #000; background:${statusColor}; color:#fff; border-radius:4px;">
            ${data.status.replace('_', ' ')}
          </span>
          <span style="font-size:12px;">Latency: ${data.latency_ms || 0}ms</span>
        </div>
        <p style="font-size:13px; margin-top:8px; color:var(--text-muted);">${data.diagnostics?.decision_reason || data.diagnostics?.reason || 'No confident match found.'}</p>
      `;
    }

    loadRecentMatches();
    loadMemoryLog();
  } catch (err) {
    showToast('Visual search request error', 'error');
  } finally {
    btn.disabled = false;
    btn.innerText = 'Run Visual Search';
  }
}

async function loadRecentMatches() {
  const tbody = document.getElementById('matchesTableBody');
  if (!tbody) return;

  try {
    const res = await fetch('/api/visual-search/recent');
    const matches = await res.json();

    if (!matches || matches.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="text-center">No visual search matches recorded yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = matches.map(m => {
      let statusColor = '#FFB800';
      if (m.match_status === 'strong_match') statusColor = '#00B865';
      if (m.match_status === 'no_match' || m.match_status === 'failed') statusColor = '#FF3333';

      const bboxStr = m.bounding_box ? `[${m.bounding_box.x}, ${m.bounding_box.y}, ${m.bounding_box.width}x${m.bounding_box.height}]` : '—';

      return `
        <tr>
          <td><code title="${m.observed_at || m.frame_timestamp || ''}">${m.formatted_time || (m.frame_timestamp ? m.frame_timestamp.split('T')[1]?.substring(0, 8) || m.frame_timestamp : '')}</code></td>
          <td><span style="font-weight:700; font-size:11px; padding:2px 6px; border:1px solid #000; background:${statusColor}; color:#fff; border-radius:3px;">${m.match_status}</span></td>
          <td><strong>${m.object_name}</strong></td>
          <td>${m.location_name || '—'}</td>
          <td><strong>${(m.similarity_score * 100).toFixed(1)}%</strong></td>
          <td><code>${bboxStr}</code></td>
          <td>${m.local_matches || 0}</td>
        </tr>
      `;
    }).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="7" class="text-center">Failed to load recent visual matches.</td></tr>`;
  }
}

/* Reality Checksum & Discrepancy Verification Controllers */
async function loadRealityChecksum() {
  // 1. Fetch summary metrics
  try {
    const sumRes = await fetch('/api/reality-checksum/summary');
    if (sumRes.ok) {
      const summary = await sumRes.json();
      const objEl = document.getElementById('realityTotalObjects');
      const actEl = document.getElementById('realityActiveConflicts');
      const relEl = document.getElementById('realityTotalRelocations');
      if (objEl) objEl.innerText = summary.total_monitored_objects || 0;
      if (actEl) actEl.innerText = summary.active_unverified_conflicts || 0;
      if (relEl) relEl.innerText = summary.total_relocations_detected || 0;
    }
  } catch (e) {
    console.error('Failed to load reality summary:', e);
  }

  // 2. Fetch verified reality states
  const statesBody = document.getElementById('realityStatesTableBody');
  if (statesBody) {
    try {
      const statesRes = await fetch('/api/reality-checksum/states');
      const states = await statesRes.json();
      if (!states || states.length === 0) {
        statesBody.innerHTML = `<tr><td colspan="6" class="text-center">No object reality states tracked yet.</td></tr>`;
      } else {
        statesBody.innerHTML = states.map(s => {
          const conflictBadge = s.has_active_conflict
            ? `<span style="font-weight:700; font-size:11px; padding:2px 6px; border:1px solid #000; background:#FFB800; color:#000; border-radius:3px;">DISCREPANCY FLAGGED</span>`
            : `<span style="font-weight:700; font-size:11px; padding:2px 6px; border:1px solid #000; background:#00B865; color:#fff; border-radius:3px;">VERIFIED</span>`;

          return `
            <tr>
              <td><strong>${s.object_name}</strong></td>
              <td><strong>${s.current_location}</strong></td>
              <td>${s.previous_location || '—'}</td>
              <td><code>${s.formatted_time || s.last_observed_at || '—'}</code></td>
              <td><span style="font-size:12px; font-weight:600;">${s.verification_status} (${s.evidence_source})</span></td>
              <td>${conflictBadge}</td>
            </tr>
          `;
        }).join('');
      }
    } catch (e) {
      statesBody.innerHTML = `<tr><td colspan="6" class="text-center">Failed to load verified reality states.</td></tr>`;
    }
  }

  // 3. Fetch conflict history & open discrepancies
  const conflictsBody = document.getElementById('realityConflictsTableBody');
  if (conflictsBody) {
    try {
      const confRes = await fetch('/api/reality-checksum/conflicts');
      const conflicts = await confRes.json();
      if (!conflicts || conflicts.length === 0) {
        conflictsBody.innerHTML = `<tr><td colspan="8" class="text-center">No discrepancies or conflicts detected.</td></tr>`;
      } else {
        conflictsBody.innerHTML = conflicts.map(c => {
          let statusColor = '#00B865';
          if (c.resolution_status === 'flagged_unverified') statusColor = '#FFB800';
          if (c.resolution_status === 'rejected') statusColor = '#FF3333';

          const actionBtn = c.resolution_status === 'flagged_unverified'
            ? `<button class="btn btn-sm" onclick="resolveDiscrepancyModal('${c.id}', '${c.object_name}', '${c.new_location}')" style="background:#00B865; color:#fff;">Verify</button>`
            : `<span style="font-size:11px; color:var(--text-muted);">✓ Resolved</span>`;

          return `
            <tr>
              <td><code>${c.formatted_detected_at || c.detected_at}</code></td>
              <td><strong>${c.object_name}</strong></td>
              <td><code>${c.conflict_type}</code></td>
              <td>${c.previous_location || '—'}</td>
              <td><strong>${c.new_location}</strong></td>
              <td><span style="font-weight:700; font-size:11px; padding:2px 6px; border:1px solid #000; background:${statusColor}; color:#fff; border-radius:3px;">${c.resolution_status}</span></td>
              <td style="font-size:12px; max-width:250px;">${c.resolution_notes || ''}</td>
              <td>${actionBtn}</td>
            </tr>
          `;
        }).join('');
      }
    } catch (e) {
      conflictsBody.innerHTML = `<tr><td colspan="8" class="text-center">Failed to load conflicts log.</td></tr>`;
    }
  }
}

async function resolveDiscrepancyModal(conflictId, objectName, candidateLoc) {
  const verifiedLoc = prompt(`Verify true location for '${objectName}':`, candidateLoc);
  if (!verifiedLoc || !verifiedLoc.trim()) return;

  try {
    const res = await fetch('/api/reality-checksum/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        conflict_id: conflictId,
        verified_location: verifiedLoc.trim(),
        verified_by: 'caregiver',
      }),
    });
    if (res.ok) {
      showToast(`Verified '${objectName}' at '${verifiedLoc.trim()}'!`);
      loadRealityChecksum();
    } else {
      const err = await res.json();
      showToast(err.detail || 'Failed to verify discrepancy', 'error');
    }
  } catch (e) {
    showToast('Network error while verifying discrepancy', 'error');
  }
}

/* ========================================================================= */
/* Laptop Voice Assistant (Mic Input & Speaker Output)                      */
/* ========================================================================= */
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

function initVoiceAssistant() {
  const startBtn = document.getElementById('startVoiceBtn');
  const stopBtn = document.getElementById('stopVoiceBtn');
  const pulseContainer = document.getElementById('micPulseContainer');
  const sendTextBtn = document.getElementById('sendVoiceTextBtn');
  const textInput = document.getElementById('voiceTextInput');

  if (startBtn) startBtn.addEventListener('click', startRecordingVoice);
  if (stopBtn) stopBtn.addEventListener('click', stopRecordingVoice);
  if (pulseContainer) {
    pulseContainer.addEventListener('click', () => {
      if (!isRecording) startRecordingVoice();
      else stopRecordingVoice();
    });
  }
  if (sendTextBtn) {
    sendTextBtn.addEventListener('click', () => {
      const q = textInput.value.trim();
      if (q) processVoiceQuery(q, false);
    });
  }
  if (textInput) {
    textInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const q = textInput.value.trim();
        if (q) processVoiceQuery(q, false);
      }
    });
  }
}

async function startRecordingVoice() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      stream.getTracks().forEach(track => track.stop());
      processVoiceQuery(audioBlob, true);
    };

    mediaRecorder.start();
    isRecording = true;
    updateRecordingUI(true);
    showToast('Recording... Speak clearly into your laptop mic.');
  } catch (err) {
    console.error('Mic access error:', err);
    showToast('Could not access laptop microphone. Please allow microphone permissions in your browser.', 'error');
  }
}

function stopRecordingVoice() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    updateRecordingUI(false);
  }
}

function updateRecordingUI(recording) {
  const startBtn = document.getElementById('startVoiceBtn');
  const stopBtn = document.getElementById('stopVoiceBtn');
  const pulse = document.getElementById('micPulseContainer');
  const title = document.getElementById('voiceStatusTitle');
  const desc = document.getElementById('voiceStatusDesc');

  if (recording) {
    if (startBtn) startBtn.style.display = 'none';
    if (stopBtn) stopBtn.style.display = 'inline-flex';
    if (pulse) {
      pulse.style.background = '#FEE2E2';
      pulse.style.borderColor = '#EF4444';
      pulse.style.transform = 'scale(1.1)';
      pulse.style.boxShadow = '0 0 20px rgba(239, 68, 68, 0.4)';
    }
    if (title) title.innerText = 'Listening to Laptop Mic...';
    if (desc) desc.innerText = 'Speak your question now (e.g. "Where are my keys?"). Click Stop when done.';
  } else {
    if (startBtn) startBtn.style.display = 'inline-flex';
    if (stopBtn) stopBtn.style.display = 'none';
    if (pulse) {
      pulse.style.background = '#EFF6FF';
      pulse.style.borderColor = '#2563EB';
      pulse.style.transform = 'scale(1.0)';
      pulse.style.boxShadow = 'none';
    }
    if (title) title.innerText = 'Click to Speak';
    if (desc) desc.innerText = 'Tap the microphone to start recording your voice through the laptop mic.';
  }
}

async function processVoiceQuery(inputData, isAudioBlob) {
  const emptyBox = document.getElementById('voiceResponseEmpty');
  const contentBox = document.getElementById('voiceResponseContent');
  const alsoPlayHost = document.getElementById('voiceAlsoPlayHost')?.checked ?? true;

  if (emptyBox) emptyBox.style.display = 'none';
  if (contentBox) contentBox.style.display = 'flex';

  document.getElementById('voiceRespTranscript').innerText = 'Processing audio via Whisper / STT...';
  document.getElementById('voiceRespIntent').innerText = '...';
  document.getElementById('voiceRespConfidence').innerText = '...';
  document.getElementById('voiceRespReasoning').innerText = 'Second Mind (llama3.1:latest) is thinking...';
  document.getElementById('voiceRespAnswer').innerText = 'Generating answer...';

  try {
    let res;
    const headers = {};
    if (alsoPlayHost) {
      headers['X-Speak-On-Host'] = 'true';
    }

    if (isAudioBlob) {
      const formData = new FormData();
      formData.append('audio', inputData, 'laptop_mic.wav');
      res = await fetch('/ask', {
        method: 'POST',
        headers: headers,
        body: formData,
      });
    } else {
      res = await fetch('/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...headers,
        },
        body: JSON.stringify({ transcript: inputData, device_id: 'browser_laptop' }),
      });
    }

    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}`);
    }

    // Extract debug headers
    const transcript = res.headers.get('X-Transcript') || (typeof inputData === 'string' ? inputData : 'Voice query');
    const intent = res.headers.get('X-Intent') || 'GENERAL';
    const confidence = res.headers.get('X-Confidence') || 'HIGH';
    const reasoning = res.headers.get('X-Reasoning') || 'Direct query';
    const answerText = res.headers.get('X-Answer-Text') || '';
    const formattedTime = res.headers.get('X-Formatted-Time') || '';

    document.getElementById('voiceRespTranscript').innerText = transcript;
    document.getElementById('voiceRespIntent').innerText = intent.toUpperCase();
    document.getElementById('voiceRespConfidence').innerText = confidence.toUpperCase();
    document.getElementById('voiceRespReasoning').innerText = reasoning;
    document.getElementById('voiceRespAnswer').innerText = answerText;

    const timeContainer = document.getElementById('voiceRespTimeContainer');
    const timeEl = document.getElementById('voiceRespTime');
    if (formattedTime) {
      timeContainer.style.display = 'block';
      timeEl.innerText = formattedTime;
    } else {
      timeContainer.style.display = 'none';
    }

    // Audio playback on laptop speakers via browser
    const audioBlob = await res.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const player = document.getElementById('voiceAudioPlayer');
    if (player) {
      player.src = audioUrl;
      player.play().catch(e => console.log('Autoplay blocked, user can click play:', e));
    }

    showToast(`Answered by llama3.1:latest (${intent})`);
  } catch (err) {
    console.error('Voice processing error:', err);
    showToast('Failed to process voice query: ' + err.message, 'error');
    document.getElementById('voiceRespAnswer').innerText = 'Error processing question. Please check backend status.';
  }
}



