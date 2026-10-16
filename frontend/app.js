/* Second Mind Caregiver Dashboard - Client Application Logic */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  loadHealthStatus();
  loadProfile();
  loadSchedule();
  loadMemoryLog();

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
          <td><code>${obs.timestamp || ''}</code></td>
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
