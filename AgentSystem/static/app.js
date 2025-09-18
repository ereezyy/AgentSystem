document.addEventListener('DOMContentLoaded', function() {
    // Initial data load
    refreshDashboardData();
    refreshTasks();
    refreshAgents();
    refreshScalingStatus();
    refreshAnalytics();

    // Set up periodic refresh
    const refreshInterval = localStorage.getItem('refreshInterval') || 5000;
    setInterval(function() {
        refreshDashboardData();
        refreshTasks();
        refreshAgents();
        refreshScalingStatus();
        refreshAnalytics();
    }, refreshInterval);
});

function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section-content').forEach(section => {
        section.style.display = 'none';
    });
    // Show selected section
    document.getElementById(sectionId).style.display = 'block';
    // Update active nav link
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    document.querySelector(`[onclick="showSection('${sectionId}')"]`).classList.add('active');
}

function refreshDashboardData() {
    fetch('/dashboard/data?time_range=3600')
        .then(response => response.json())
        .then(data => {
            updateSystemHealth(data);
            updateActiveTasks(data);
            updateActiveAgents(data);
            updateCharts(data);
        })
        .catch(error => console.error('Error fetching dashboard data:', error));
}

function updateSystemHealth(data) {
    const healthElement = document.getElementById('system-health');
    const cpuUsage = data.system_metrics.cpu_usage[data.system_metrics.cpu_usage.length - 1]?.value || 0;
    const memoryUsage = data.system_metrics.memory_usage[data.system_metrics.memory_usage.length - 1]?.value || 0;
    healthElement.innerHTML = `CPU: ${cpuUsage.toFixed(1)}% | Memory: ${memoryUsage.toFixed(1)}%`;
}

function updateActiveTasks(data) {
    const tasksElement = document.getElementById('active-tasks');
    const activeTasks = data.current_status.swarm.active_tasks || 0;
    tasksElement.innerHTML = `${activeTasks} active task(s)`;
}

function updateActiveAgents(data) {
    const agentsElement = document.getElementById('active-agents');
    const activeAgents = Object.keys(data.current_status.swarm.agent_status || {}).length;
    agentsElement.innerHTML = `${activeAgents} active agent(s)`;
}

function updateCharts(data) {
    // CPU Chart
    const cpuData = data.system_metrics.cpu_usage.map(point => ({
        x: new Date(point.timestamp * 1000),
        y: point.value
    }));
    createChart('cpu-chart', 'CPU Usage (%)', cpuData);

    // Memory Chart
    const memoryData = data.system_metrics.memory_usage.map(point => ({
        x: new Date(point.timestamp * 1000),
        y: point.value
    }));
    createChart('memory-chart', 'Memory Usage (%)', memoryData);

    // Response Time Chart
    const responseTimeData = data.performance_metrics.response_time.map(point => ({
        x: new Date(point.timestamp * 1000),
        y: point.value
    }));
    createChart('response-time-chart', 'Response Time (s)', responseTimeData);
}

function createChart(elementId, label, data) {
    const ctx = document.getElementById(elementId).getContext('2d');
    if (window[elementId]) {
        window[elementId].destroy(); // Destroy existing chart if it exists
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
                    time: {
                        unit: 'minute'
                    },
                    title: {
                        display: true,
                        text: 'Time'
                    }
                },
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: label
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

function refreshTasks() {
    fetch('/tasks')
        .then(response => response.json())
        .then(tasks => {
            const taskList = document.getElementById('task-list');
            taskList.innerHTML = '';
            tasks.forEach(task => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${task.task_id}</td>
                    <td>${task.name}</td>
                    <td>${task.description}</td>
                    <td>${task.status}</td>
                    <td>
                        <button class="btn btn-sm btn-primary" onclick="startTask('${task.task_id}')">Start</button>
                        <button class="btn btn-sm btn-danger" onclick="stopTask('${task.task_id}')">Stop</button>
                    </td>
                `;
                taskList.appendChild(row);
            });
        })
        .catch(error => console.error('Error fetching tasks:', error));
}

function startTask(taskId) {
    fetch(`/task/${taskId}/start`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started') {
                alert(`Task ${taskId} started`);
                refreshTasks();
            } else {
                alert(`Failed to start task: ${data.detail}`);
            }
        })
        .catch(error => console.error('Error starting task:', error));
}

function stopTask(taskId) {
    fetch(`/task/${taskId}/stop`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'stopped') {
                alert(`Task ${taskId} stopped`);
                refreshTasks();
            } else {
                alert(`Failed to stop task: ${data.detail}`);
            }
        })
        .catch(error => console.error('Error stopping task:', error));
}

function refreshAgents() {
    fetch('/swarm/status')
        .then(response => response.json())
        .then(status => {
            const agentList = document.getElementById('agent-list');
            agentList.innerHTML = '';
            for (const [agentId, agentInfo] of Object.entries(status.agent_status || {})) {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${agentId}</td>
                    <td>${agentInfo.role}</td>
                    <td>${agentInfo.status}</td>
                    <td>${agentInfo.last_activity || 'N/A'}</td>
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
                        <button class="btn btn-sm btn-primary" onclick="manualScale('${serviceName}', ${serviceInfo.current_instances + 1})">+</button>
                        <button class="btn btn-sm btn-danger" onclick="manualScale('${serviceName}', ${serviceInfo.current_instances - 1})">-</button>
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
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            service_name: serviceName,
            target_instances: targetInstances
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert(`Scaled ${serviceName} to ${targetInstances} instances`);
            refreshScalingStatus();
        } else {
            alert(`Failed to scale ${serviceName}`);
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
                item.innerHTML = `<strong>${insight.title}</strong>: ${insight.description} (Impact: ${insight.impact})`;
                insightsList.appendChild(item);
            });
        })
        .catch(error => console.error('Error fetching insights:', error));
}

function openTaskModal() {
    const modal = new bootstrap.Modal(document.getElementById('taskModal'));
    modal.show();
}

function closeTaskModal() {
    const modal = bootstrap.Modal.getInstance(document.getElementById('taskModal'));
    modal.hide();
}

function submitTask() {
    const name = document.getElementById('task-name').value;
    const description = document.getElementById('task-description').value;
    const priority = parseInt(document.getElementById('task-priority').value);

    fetch('/task', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            task: description,
            priority: priority
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'started') {
            alert(`Task ${name} submitted with ID ${data.task_id}`);
            closeTaskModal();
            refreshTasks();
        } else {
            alert(`Failed to submit task`);
        }
    })
    .catch(error => console.error('Error submitting task:', error));
}

function saveSettings() {
    const interval = parseInt(document.getElementById('refresh-interval').value) * 1000;
    localStorage.setItem('refreshInterval', interval);
    alert('Settings saved. Refresh interval updated to ' + interval/1000 + ' seconds.');
    location.reload(); // Reload to apply new refresh interval
}
