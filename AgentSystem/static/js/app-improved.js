document.addEventListener('DOMContentLoaded', function() {
  refreshDashboardData();
  refreshTasks();
  refreshAgents();
  refreshScalingStatus();
  refreshAnalytics();

  const refreshInterval = localStorage.getItem('refreshInterval') || 5000;
  setInterval(function() {
    refreshDashboardData();
    refreshTasks();
    refreshAgents();
    refreshScalingStatus();
    refreshAnalytics();
  }, refreshInterval);

  initializeCharacterCounters();
  initializeFormValidation();
});

function showSection(sectionId) {
  document.querySelectorAll('.section-content').forEach(section => {
    section.style.display = 'none';
  });
  document.getElementById(sectionId).style.display = 'block';

  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.remove('active');
  });
  document.querySelector(`[onclick="showSection('${sectionId}')"]`).classList.add('active');
}

function refreshDashboardData() {
  showSkeletonLoader('system-health');
  showSkeletonLoader('active-tasks');
  showSkeletonLoader('active-agents');

  fetch('/dashboard/data?time_range=3600')
    .then(response => response.json())
    .then(data => {
      updateSystemHealth(data);
      updateActiveTasks(data);
      updateActiveAgents(data);
      updateCharts(data);
    })
    .catch(error => {
      console.error('Error fetching dashboard data:', error);
      showBanner({
        type: 'warning',
        message: 'We\'re having trouble loading your dashboard',
        help: 'Check your internet connection and we\'ll keep trying.',
        dismissible: false
      });
      scheduleRetry(() => refreshDashboardData(), 5000);
    });
}

function updateSystemHealth(data) {
  const healthElement = document.getElementById('system-health');
  const cpuUsage = data.system_metrics.cpu_usage[data.system_metrics.cpu_usage.length - 1]?.value || 0;
  const memoryUsage = data.system_metrics.memory_usage[data.system_metrics.memory_usage.length - 1]?.value || 0;

  const cpuStatus = cpuUsage < 70 ? 'status-good' : cpuUsage < 85 ? 'status-warning' : 'status-danger';
  const memStatus = memoryUsage < 70 ? 'status-good' : memoryUsage < 85 ? 'status-warning' : 'status-danger';

  healthElement.innerHTML = `
    <div class="metric-group">
      <div class="metric">
        <span class="metric-label">CPU Usage</span>
        <span class="metric-value ${cpuStatus}">${cpuUsage.toFixed(1)}%</span>
        <span class="metric-status">${cpuUsage < 70 ? 'Normal' : cpuUsage < 85 ? 'Elevated' : 'High'}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Memory Usage</span>
        <span class="metric-value ${memStatus}">${memoryUsage.toFixed(1)}%</span>
        <span class="metric-status">${memoryUsage < 70 ? 'Normal' : memoryUsage < 85 ? 'Elevated' : 'High'}</span>
      </div>
    </div>
  `;
}

function updateActiveTasks(data) {
  const tasksElement = document.getElementById('active-tasks');
  const activeTasks = data.current_status.swarm.active_tasks || 0;
  tasksElement.innerHTML = `
    <div class="metric">
      <span class="metric-value">${activeTasks}</span>
      <span class="metric-label">${activeTasks === 1 ? 'task running' : 'tasks running'}</span>
    </div>
  `;
}

function updateActiveAgents(data) {
  const agentsElement = document.getElementById('active-agents');
  const activeAgents = Object.keys(data.current_status.swarm.agent_status || {}).length;
  agentsElement.innerHTML = `
    <div class="metric">
      <span class="metric-value">${activeAgents}</span>
      <span class="metric-label">${activeAgents === 1 ? 'agent working' : 'agents working'}</span>
    </div>
  `;
}

function updateCharts(data) {
  const cpuData = data.system_metrics.cpu_usage.map(point => ({
    x: new Date(point.timestamp * 1000),
    y: point.value
  }));
  createChart('cpu-chart', 'CPU Usage (%)', cpuData);

  const memoryData = data.system_metrics.memory_usage.map(point => ({
    x: new Date(point.timestamp * 1000),
    y: point.value
  }));
  createChart('memory-chart', 'Memory Usage (%)', memoryData);

  const responseTimeData = data.performance_metrics.response_time.map(point => ({
    x: new Date(point.timestamp * 1000),
    y: point.value
  }));
  createChart('response-time-chart', 'Response Time (s)', responseTimeData);
}

function createChart(elementId, label, data) {
  const ctx = document.getElementById(elementId).getContext('2d');
  if (window[elementId]) {
    window[elementId].destroy();
  }
  window[elementId] = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [{
        label: label,
        data: data,
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: {
          type: 'time',
          time: { unit: 'minute' },
          title: { display: true, text: 'Time' }
        },
        y: {
          beginAtZero: true,
          title: { display: true, text: label }
        }
      },
      plugins: { legend: { display: false } }
    }
  });
}

function refreshTasks() {
  const taskList = document.getElementById('task-list');
  showSkeletonLoader('task-list');

  fetch('/tasks')
    .then(response => response.json())
    .then(tasks => {
      if (tasks.length === 0) {
        taskList.innerHTML = `
          <tr>
            <td colspan="5">
              <div class="empty-state">
                <div class="empty-state-icon">📋</div>
                <h3>No tasks yet</h3>
                <p>Create your first task to get your AI agents working</p>
                <button class="btn btn-primary" onclick="openTaskModal()">
                  Create Your First Task
                </button>
              </div>
            </td>
          </tr>
        `;
        return;
      }

      taskList.innerHTML = '';
      tasks.forEach(task => {
        const statusBadge = getStatusBadge(task.status);
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${task.task_id}</td>
          <td>${task.name}</td>
          <td>${task.description}</td>
          <td>${statusBadge}</td>
          <td>
            ${task.status === 'PENDING' ? `
              <button class="btn btn-sm btn-primary"
                      onclick="startTask('${task.task_id}')"
                      data-tooltip="Begin working on this task">
                ▶ Resume
              </button>
            ` : ''}
            ${task.status === 'RUNNING' ? `
              <button class="btn btn-sm btn-warning"
                      onclick="confirmPauseTask('${task.task_id}')"
                      data-tooltip="Pause this task">
                ⏸ Pause
              </button>
            ` : ''}
            <button class="btn btn-sm btn-danger"
                    onclick="confirmDeleteTask('${task.task_id}', '${task.name}')"
                    data-tooltip="Delete this task permanently">
              Delete
            </button>
          </td>
        `;
        taskList.appendChild(row);
      });
    })
    .catch(error => {
      console.error('Error fetching tasks:', error);
      showToast({
        type: 'error',
        title: 'Couldn\'t Load Tasks',
        message: 'We\'re having trouble loading your tasks. Please try again.',
        duration: 5000
      });
    });
}

function getStatusBadge(status) {
  const badges = {
    'RUNNING': '<span class="status-badge status-active">🟢 Active</span>',
    'PENDING': '<span class="status-badge status-waiting">⏳ Waiting</span>',
    'COMPLETED': '<span class="status-badge status-completed">✅ Done</span>',
    'STOPPED': '<span class="status-badge status-paused">⏸ Paused</span>',
    'FAILED': '<span class="status-badge status-failed">⚠️ Failed</span>'
  };
  return badges[status] || `<span class="status-badge">${status}</span>`;
}

function startTask(taskId) {
  microcopyManager.trackInteraction('task_start_button', 'click');

  fetch(`/task/${taskId}/start`, { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      if (data.status === 'started') {
        microcopyManager.trackInteraction('task_start_button', 'click', 'success');
        showToast({
          type: 'success',
          title: 'Task Started',
          message: 'Your AI agent is working on it now',
          icon: '✓',
          duration: 3000
        });
        refreshTasks();
      } else {
        microcopyManager.trackInteraction('task_start_button', 'click', 'error');
        showToast({
          type: 'error',
          title: 'Couldn\'t Start Task',
          message: data.help || data.detail || 'Something went wrong. Please try again.',
          duration: 5000
        });
      }
    })
    .catch(error => {
      console.error('Error starting task:', error);
      showToast({
        type: 'error',
        title: 'Connection Error',
        message: 'Check your internet connection and try again.',
        duration: 5000
      });
    });
}

function confirmPauseTask(taskId) {
  showModal({
    title: 'Pause this task?',
    message: 'The AI agent will stop working on this task. You can resume it anytime.',
    type: 'warning',
    icon: '⏸',
    actions: [
      {
        label: 'Keep Running',
        style: 'secondary',
        onClick: () => {}
      },
      {
        label: 'Pause Task',
        style: 'warning',
        onClick: () => stopTask(taskId)
      }
    ]
  });
}

function stopTask(taskId) {
  fetch(`/task/${taskId}/stop`, { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      if (data.status === 'stopped') {
        showToast({
          type: 'success',
          title: 'Task Paused',
          message: 'You can resume this task anytime',
          duration: 3000
        });
        refreshTasks();
      } else {
        showToast({
          type: 'error',
          title: 'Couldn\'t Pause Task',
          message: data.help || data.detail || 'Please try again.',
          duration: 5000
        });
      }
    })
    .catch(error => console.error('Error stopping task:', error));
}

function confirmDeleteTask(taskId, taskName) {
  showModal({
    title: 'Delete this task?',
    message: `"${taskName}" and all its data will be permanently deleted. This can't be undone.`,
    type: 'danger',
    icon: '⚠️',
    actions: [
      {
        label: 'Cancel',
        style: 'secondary',
        onClick: () => {}
      },
      {
        label: 'Delete Permanently',
        style: 'danger',
        onClick: () => deleteTask(taskId)
      }
    ]
  });
}

function deleteTask(taskId) {
  fetch(`/task/${taskId}/delete`, { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      if (data.status === 'deleted') {
        showToast({
          type: 'success',
          title: 'Task Deleted',
          message: 'The task has been permanently removed',
          duration: 3000
        });
        refreshTasks();
      }
    })
    .catch(error => console.error('Error deleting task:', error));
}

function refreshAgents() {
  fetch('/swarm/status')
    .then(response => response.json())
    .then(status => {
      const agentList = document.getElementById('agent-list');

      if (Object.keys(status.agent_status || {}).length === 0) {
        agentList.innerHTML = `
          <tr>
            <td colspan="4">
              <div class="empty-state">
                <div class="empty-state-icon">🤖</div>
                <h3>All agents are idle</h3>
                <p>Your agents will appear here when they're working on tasks</p>
                <a href="#" onclick="showSection('tasks')" class="text-primary">
                  View your tasks →
                </a>
              </div>
            </td>
          </tr>
        `;
        return;
      }

      agentList.innerHTML = '';
      for (const [agentId, agentInfo] of Object.entries(status.agent_status || {})) {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${agentId}</td>
          <td>${agentInfo.role}</td>
          <td>${getStatusBadge(agentInfo.status)}</td>
          <td>${agentInfo.last_activity || 'Just now'}</td>
        `;
        agentList.appendChild(row);
      }
    })
    .catch(error => console.error('Error fetching agents:', error));
}

function refreshScalingStatus() {
  fetch('/scaling/status')
    .then(response => response.json())
    .then(status => {
      const scalingStatus = document.getElementById('scaling-status');
      scalingStatus.innerHTML = '';
      for (const [serviceName, serviceInfo] of Object.entries(status.services || {})) {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td>${serviceName}</td>
          <td>${serviceInfo.current_instances}</td>
          <td>${serviceInfo.min_instances}</td>
          <td>${serviceInfo.max_instances}</td>
          <td>${serviceInfo.scaling_metric}</td>
          <td>
            <button class="btn btn-sm btn-primary"
                    onclick="manualScale('${serviceName}', ${serviceInfo.current_instances + 1})"
                    data-tooltip="Add one more instance">
              + Scale Up
            </button>
            <button class="btn btn-sm btn-secondary"
                    onclick="manualScale('${serviceName}', ${serviceInfo.current_instances - 1})"
                    data-tooltip="Remove one instance"
                    ${serviceInfo.current_instances <= serviceInfo.min_instances ? 'disabled' : ''}>
              - Scale Down
            </button>
          </td>
        `;
        scalingStatus.appendChild(row);
      }
    })
    .catch(error => console.error('Error fetching scaling status:', error));
}

function manualScale(serviceName, targetInstances) {
  fetch('/scaling/manual', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      service_name: serviceName,
      target_instances: targetInstances
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      showToast({
        type: 'success',
        title: 'Scaling Updated',
        message: `${serviceName} now has ${targetInstances} instance${targetInstances === 1 ? '' : 's'}`,
        duration: 3000
      });
      refreshScalingStatus();
    } else {
      showToast({
        type: 'error',
        title: 'Scaling Failed',
        message: 'Unable to scale this service. Please try again.',
        duration: 5000
      });
    }
  })
  .catch(error => console.error('Error scaling service:', error));
}

function refreshAnalytics() {
  fetch('/analytics/insights?time_range=86400')
    .then(response => response.json())
    .then(data => {
      const insightsList = document.getElementById('insights-list');
      insightsList.innerHTML = '';
      data.insights.forEach(insight => {
        const item = document.createElement('li');
        item.className = 'list-group-item';
        item.innerHTML = `
          <div class="d-flex justify-content-between align-items-start">
            <div class="flex-grow-1">
              <strong>${insight.title}</strong>
              <p class="mb-0 mt-1 text-muted">${insight.description}</p>
            </div>
            <span class="badge bg-${insight.impact === 'high' ? 'danger' : insight.impact === 'medium' ? 'warning' : 'info'}">
              ${insight.impact} impact
            </span>
          </div>
        `;
        insightsList.appendChild(item);
      });
    })
    .catch(error => console.error('Error fetching insights:', error));
}

function openTaskModal() {
  microcopyManager.trackInteraction('task_create_button', 'click');
  const modal = new bootstrap.Modal(document.getElementById('taskModal'));
  modal.show();
}

function closeTaskModal() {
  const modal = bootstrap.Modal.getInstance(document.getElementById('taskModal'));
  modal.hide();
}

function submitTask() {
  microcopyManager.trackInteraction('task_submit_button', 'click');

  const name = document.getElementById('task-name').value.trim();
  const description = document.getElementById('task-description').value.trim();
  const priority = parseInt(document.getElementById('task-priority').value);

  if (!name || !description) {
    showToast({
      type: 'error',
      title: 'Missing Information',
      message: 'Please fill in both the task name and description',
      duration: 5000
    });
    return;
  }

  fetch('/task', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      task: description,
      priority: priority
    })
  })
  .then(response => response.json())
  .then(data => {
    if (data.status === 'started') {
      microcopyManager.trackInteraction('task_submit_button', 'click', 'success');
      showToast({
        type: 'success',
        title: 'Task Created Successfully',
        message: `"${name}" is now in your task queue`,
        help: 'It will start automatically based on priority.',
        action: {
          label: 'Track Progress',
          onClick: () => showSection('tasks')
        },
        duration: 5000
      });
      closeTaskModal();
      document.getElementById('task-form').reset();
      refreshTasks();
    } else {
      microcopyManager.trackInteraction('task_submit_button', 'click', 'error');
      showToast({
        type: 'error',
        title: 'Couldn\'t Create Task',
        message: 'Something went wrong. Please try again.',
        duration: 5000
      });
    }
  })
  .catch(error => {
    console.error('Error submitting task:', error);
    showToast({
      type: 'error',
      title: 'Connection Error',
      message: 'Check your internet connection and try again.',
      duration: 5000
    });
  });
}

function saveSettings() {
  const interval = parseInt(document.getElementById('refresh-interval').value) * 1000;
  localStorage.setItem('refreshInterval', interval);
  showToast({
    type: 'success',
    title: 'Settings Saved',
    message: `Dashboard will now update every ${interval/1000} second${interval === 1000 ? '' : 's'}`,
    icon: '✓',
    duration: 3000
  });
  setTimeout(() => location.reload(), 1000);
}

function showSkeletonLoader(elementId) {
  const element = document.getElementById(elementId);
  if (!element) return;
  element.innerHTML = `
    <div class="skeleton-loader">
      <div class="skeleton-line" style="width: 60%"></div>
      <div class="skeleton-line" style="width: 80%"></div>
      <div class="skeleton-line" style="width: 45%"></div>
    </div>
  `;
}

function scheduleRetry(callback, delay) {
  setTimeout(callback, delay);
}

function initializeCharacterCounters() {
  const textarea = document.getElementById('task-description');
  if (textarea) {
    const counter = document.getElementById('char-count');
    textarea.addEventListener('input', () => {
      const length = textarea.value.length;
      const maxLength = textarea.getAttribute('maxlength');
      if (counter) {
        counter.textContent = length;
        if (length > maxLength * 0.9) {
          counter.parentElement.classList.add('near-limit');
        } else {
          counter.parentElement.classList.remove('near-limit');
        }
        if (length >= maxLength) {
          counter.parentElement.classList.add('at-limit');
        } else {
          counter.parentElement.classList.remove('at-limit');
        }
      }
    });
  }
}

function initializeFormValidation() {
  const forms = document.querySelectorAll('form');
  forms.forEach(form => {
    form.addEventListener('submit', (e) => {
      if (!form.checkValidity()) {
        e.preventDefault();
        e.stopPropagation();
        form.classList.add('was-validated');
      }
    });
  });
}
