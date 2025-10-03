class MicrocopyManager {
  constructor() {
    this.variants = new Map();
    this.sessionId = this.generateSessionId();
    this.currentVariants = new Map();
  }

  generateSessionId() {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  async loadVariants() {
    try {
      const response = await fetch('/api/microcopy/variants');
      const data = await response.json();

      data.variants.forEach(variant => {
        if (!this.variants.has(variant.key)) {
          this.variants.set(variant.key, []);
        }
        this.variants.get(variant.key).push(variant);
      });

      this.assignVariants();
    } catch (error) {
      console.error('Failed to load microcopy variants:', error);
    }
  }

  assignVariants() {
    this.variants.forEach((variants, key) => {
      if (variants.length === 0) return;

      const savedVariant = localStorage.getItem(`microcopy_${key}`);
      let selectedVariant;

      if (savedVariant) {
        selectedVariant = variants.find(v => v.variant_name === savedVariant);
      }

      if (!selectedVariant) {
        selectedVariant = variants[Math.floor(Math.random() * variants.length)];
        localStorage.setItem(`microcopy_${key}`, selectedVariant.variant_name);
      }

      this.currentVariants.set(key, selectedVariant);
    });
  }

  get(key) {
    return this.currentVariants.get(key)?.content || null;
  }

  async trackInteraction(key, interactionType, outcome = null, metadata = {}) {
    const variant = this.currentVariants.get(key);
    if (!variant) return;

    try {
      await fetch('/api/microcopy/interactions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          variant_id: variant.id,
          session_id: this.sessionId,
          interaction_type: interactionType,
          outcome: outcome,
          metadata: metadata
        })
      });
    } catch (error) {
      console.error('Failed to track microcopy interaction:', error);
    }
  }

  applyToElement(elementId, key) {
    const content = this.get(key);
    if (!content) return;

    const element = document.getElementById(elementId);
    if (!element) return;

    if (content.text) {
      element.textContent = content.text;
    }

    if (content.tooltip) {
      element.setAttribute('title', content.tooltip);
      element.setAttribute('data-tooltip', content.tooltip);
    }

    if (content.ariaLabel) {
      element.setAttribute('aria-label', content.ariaLabel);
    }

    if (content.placeholder && element.tagName === 'INPUT' || element.tagName === 'TEXTAREA') {
      element.setAttribute('placeholder', content.placeholder);
    }
  }
}

class ToastNotification {
  constructor() {
    this.container = this.createContainer();
  }

  createContainer() {
    let container = document.getElementById('toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      container.className = 'toast-container position-fixed top-0 end-0 p-3';
      container.style.zIndex = '9999';
      document.body.appendChild(container);
    }
    return container;
  }

  show({ type = 'info', title, message, help, action, duration = 5000, icon }) {
    const toast = document.createElement('div');
    toast.className = `toast align-items-center border-0 toast-${type}`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');

    const iconMap = {
      success: '✓',
      error: '✕',
      warning: '⚠',
      info: 'ℹ'
    };

    const displayIcon = icon || iconMap[type] || '';

    let actionHtml = '';
    if (action) {
      actionHtml = `
        <button class="btn btn-sm btn-link text-white text-decoration-none"
                onclick="${action.onClick ? action.onClick.toString() : `window.location.href='${action.url}'`}">
          ${action.label}
        </button>
      `;
    }

    toast.innerHTML = `
      <div class="toast-header bg-${type === 'error' ? 'danger' : type === 'success' ? 'success' : type === 'warning' ? 'warning' : 'primary'} text-white">
        <span class="me-2">${displayIcon}</span>
        <strong class="me-auto">${title}</strong>
        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
      </div>
      <div class="toast-body">
        ${message}
        ${help ? `<div class="mt-2 small text-muted">${help}</div>` : ''}
        ${actionHtml}
      </div>
    `;

    this.container.appendChild(toast);

    const bsToast = new bootstrap.Toast(toast, {
      autohide: duration > 0,
      delay: duration
    });

    bsToast.show();

    toast.addEventListener('hidden.bs.toast', () => {
      toast.remove();
    });

    return toast;
  }
}

class ModalManager {
  show({ title, message, type = 'info', icon, actions = [] }) {
    const existingModal = document.getElementById('dynamic-modal');
    if (existingModal) {
      existingModal.remove();
    }

    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'dynamic-modal';
    modal.setAttribute('tabindex', '-1');
    modal.setAttribute('role', 'dialog');

    const typeColors = {
      danger: 'danger',
      warning: 'warning',
      success: 'success',
      info: 'primary'
    };

    const headerColor = typeColors[type] || 'primary';

    const actionsHtml = actions.map((action, index) => `
      <button type="button"
              class="btn btn-${action.style || 'secondary'}"
              data-action-index="${index}">
        ${action.label}
      </button>
    `).join('');

    modal.innerHTML = `
      <div class="modal-dialog modal-dialog-centered">
        <div class="modal-content">
          <div class="modal-header bg-${headerColor} text-white">
            ${icon ? `<span class="me-2 fs-4">${icon}</span>` : ''}
            <h5 class="modal-title">${title}</h5>
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body">
            <p class="mb-0">${message}</p>
          </div>
          <div class="modal-footer">
            ${actionsHtml}
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(modal);

    actions.forEach((action, index) => {
      const button = modal.querySelector(`[data-action-index="${index}"]`);
      if (button && action.onClick) {
        button.addEventListener('click', () => {
          action.onClick();
          const bsModal = bootstrap.Modal.getInstance(modal);
          if (bsModal) bsModal.hide();
        });
      }
    });

    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();

    modal.addEventListener('hidden.bs.modal', () => {
      modal.remove();
    });

    return modal;
  }
}

class BannerManager {
  show({ type = 'info', message, help, dismissible = true }) {
    const existingBanner = document.getElementById('global-banner');
    if (existingBanner) {
      existingBanner.remove();
    }

    const banner = document.createElement('div');
    banner.id = 'global-banner';
    banner.className = `alert alert-${type} alert-dismissible fade show mb-0`;
    banner.setAttribute('role', 'alert');
    banner.style.borderRadius = '0';

    banner.innerHTML = `
      <div class="container-fluid">
        <div class="d-flex align-items-center">
          <div class="flex-grow-1">
            <strong>${message}</strong>
            ${help ? `<div class="small mt-1">${help}</div>` : ''}
          </div>
          ${dismissible ? '<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>' : ''}
        </div>
      </div>
    `;

    document.body.insertBefore(banner, document.body.firstChild);

    return banner;
  }
}

const microcopyManager = new MicrocopyManager();
const toastNotification = new ToastNotification();
const modalManager = new ModalManager();
const bannerManager = new BannerManager();

function showToast(options) {
  return toastNotification.show(options);
}

function showModal(options) {
  return modalManager.show(options);
}

function showBanner(options) {
  return bannerManager.show(options);
}

document.addEventListener('DOMContentLoaded', async () => {
  await microcopyManager.loadVariants();

  microcopyManager.applyToElement('create-task-btn', 'task_create_button');
  microcopyManager.applyToElement('submit-task-btn', 'task_submit_button');
  microcopyManager.applyToElement('task-name-label', 'form_task_name');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const key = entry.target.getAttribute('data-microcopy-key');
        if (key) {
          microcopyManager.trackInteraction(key, 'view');
        }
      }
    });
  });

  document.querySelectorAll('[data-microcopy-key]').forEach(element => {
    observer.observe(element);
  });
});
