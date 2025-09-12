document.addEventListener('DOMContentLoaded', function() {
    // Fetch agent information and status on page load
    fetchAgentInfo();

    // Set up event listeners for buttons
    document.getElementById('submitTask').addEventListener('click', submitTask);
    document.getElementById('navigateBrowser').addEventListener('click', navigateBrowser);
    document.getElementById('sendEmail').addEventListener('click', sendEmail);
    document.getElementById('refreshTasks').addEventListener('click', updateTasksList);
    
    // Set up port selector
    document.getElementById('portSelector').addEventListener('change', function() {
        updateBaseUrl(this.value);
    });

    // Periodically update task list and status
    setInterval(updateTasksList, 3000); // Update every 3 seconds for more real-time feel
    setInterval(fetchAgentInfo, 10000); // Update agent info every 10 seconds
    
    // Initialize port selector with possible ports
    initializePortSelector();
});

let baseUrl = window.location.origin;

function initializePortSelector() {
    const selector = document.getElementById('portSelector');
    const ports = [8080, 8081, 8082, 8083];
    ports.forEach(port => {
        const option = document.createElement('option');
        option.value = port;
        option.textContent = `Port ${port}`;
        if (window.location.port == port) {
            option.selected = true;
            baseUrl = `http://${window.location.hostname}:${port}`;
        }
        selector.appendChild(option);
    });
}

function updateBaseUrl(port) {
    baseUrl = `http://${window.location.hostname}:${port}`;
    fetchAgentInfo();
    updateTasksList();
}

function fetchWithBaseUrl(endpoint) {
    return fetch(`${baseUrl}${endpoint}`);
}

function postWithBaseUrl(endpoint, data) {
    return fetch(`${baseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });
}

function fetchAgentInfo() {
    fetchWithBaseUrl('/')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            document.getElementById('agentDescription').textContent = data.description;
            document.getElementById('status').textContent = data.status;
            document.getElementById('capabilities').textContent = data.capabilities.join(', ');

            // Show or hide browser and email controls based on capabilities
            const browserSection = document.getElementById('browserControl');
            const emailSection = document.getElementById('emailControl');
            
            if (data.capabilities.includes('browser')) {
                browserSection.classList.remove('hidden');
            } else {
                browserSection.classList.add('hidden');
            }
            
            if (data.capabilities.includes('email')) {
                emailSection.classList.remove('hidden');
            } else {
                emailSection.classList.add('hidden');
            }
        })
        .catch(error => {
            console.error('Error fetching agent info:', error);
            document.getElementById('status').textContent = 'Error fetching status';
        });
}

function submitTask() {
    const taskText = document.getElementById('taskInput').value.trim();
    const priority = parseInt(document.getElementById('priorityInput').value);

    if (!taskText) {
        alert('Please enter a task description');
        return;
    }

    const taskData = {
        task: taskText,
        priority: priority
    };

    postWithBaseUrl('/task', taskData)
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        alert(`Task submitted successfully. Task ID: ${data.task_id}`);
        document.getElementById('taskInput').value = '';
        updateTasksList();
    })
    .catch(error => {
        console.error('Error submitting task:', error);
        alert('Failed to submit task. Check console for details.');
    });
}

function updateTasksList() {
    const tasksList = document.getElementById('tasksList');
    tasksList.innerHTML = '<p>Loading tasks...</p>';
    document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
    
    fetchWithBaseUrl('/tasks')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(tasks => {
            tasksList.innerHTML = '';
            
            if (tasks.length === 0) {
                tasksList.innerHTML = '<p>No tasks available. Submit a new task to get started.</p>';
                return;
            }
            
            tasks.forEach(task => {
                const taskElement = document.createElement('div');
                taskElement.className = 'task-item';
                let statusClass = 'status-pending';
                if (task.status.toLowerCase() === 'running') {
                    statusClass = 'status-running';
                } else if (task.status.toLowerCase() === 'completed') {
                    statusClass = 'status-completed';
                } else if (task.status.toLowerCase() === 'failed') {
                    statusClass = 'status-failed';
                }
                
                taskElement.innerHTML = `
                    <h4>${task.name}: ${task.description}</h4>
                    <p>Status: <span class="task-status ${statusClass}">${task.status}</span></p>
                    <p>Created: ${new Date(task.created_at).toLocaleString()}</p>
                    ${task.started_at ? `<p>Started: ${new Date(task.started_at).toLocaleString()}</p>` : ''}
                    ${task.completed_at ? `<p>Completed: ${new Date(task.completed_at).toLocaleString()}</p>` : ''}
                    ${task.progress ? `<p>Progress: ${task.progress}%</p>` : ''}
                    <div class="task-controls">
                        <button onclick="startTask('${task.task_id}')" ${task.status.toLowerCase() === 'running' || task.status.toLowerCase() === 'completed' ? 'disabled' : ''}>Start</button>
                        <button onclick="stopTask('${task.task_id}')" ${task.status.toLowerCase() !== 'running' ? 'disabled' : ''}>Stop</button>
                        <button onclick="deleteTask('${task.task_id}')">Delete</button>
                    </div>
                `;
                tasksList.appendChild(taskElement);
            });
        })
        .catch(error => {
            console.error('Error fetching tasks:', error);
            tasksList.innerHTML = '<p>Error loading tasks. Please try again later.</p>';
        });
}

function startTask(taskId) {
    postWithBaseUrl(`/task/${taskId}/start`, {})
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        alert(`Task ${taskId} started.`);
        updateTasksList();
    })
    .catch(error => {
        console.error(`Error starting task ${taskId}:`, error);
        alert(`Failed to start task ${taskId}. Check console for details.`);
    });
}

function stopTask(taskId) {
    postWithBaseUrl(`/task/${taskId}/stop`, {})
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        alert(`Task ${taskId} stopped.`);
        updateTasksList();
    })
    .catch(error => {
        console.error(`Error stopping task ${taskId}:`, error);
        alert(`Failed to stop task ${taskId}. Check console for details.`);
    });
}

function deleteTask(taskId) {
    if (!confirm(`Are you sure you want to delete task ${taskId}?`)) {
        return;
    }
    postWithBaseUrl(`/task/${taskId}/delete`, {})
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        alert(`Task ${taskId} deleted.`);
        updateTasksList();
    })
    .catch(error => {
        console.error(`Error deleting task ${taskId}:`, error);
        alert(`Failed to delete task ${taskId}. Check console for details.`);
    });
}

function navigateBrowser() {
    const url = document.getElementById('urlInput').value.trim();

    if (!url) {
        alert('Please enter a URL');
        return;
    }

    postWithBaseUrl('/browser/navigate', { url: url })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            alert(`Browser navigated to ${data.url}`);
        } else {
            alert(`Failed to navigate: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Error navigating browser:', error);
        alert('Failed to navigate browser. Check console for details.');
    });
}

function sendEmail() {
    const to = document.getElementById('toInput').value.trim();
    const subject = document.getElementById('subjectInput').value.trim();
    const body = document.getElementById('bodyInput').value.trim();

    if (!to || !subject || !body) {
        alert('Please fill in all email fields');
        return;
    }

    const emailData = {
        to: to,
        subject: subject,
        body: body
    };

    postWithBaseUrl('/email/send', emailData)
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            alert('Email sent successfully');
            document.getElementById('toInput').value = '';
            document.getElementById('subjectInput').value = '';
            document.getElementById('bodyInput').value = '';
        } else {
            alert('Failed to send email');
        }
    })
    .catch(error => {
        console.error('Error sending email:', error);
        alert('Failed to send email. Check console for details.');
    });
}