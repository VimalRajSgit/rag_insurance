/**
 * Auto Insurance Analysis App - Frontend Logic
 * Author: RAG (Retrieval-Augmented Generation) Bot
 * Description: Enhances Streamlit UI with form validation and loading feedback
 * for a professional auto insurance analysis interface.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Streamlit dynamically generates form IDs, so we target by class or structure
    const form = document.querySelector('form');
    const submitButton = form?.querySelector('button[kind="formSubmit"]');
    const textArea = form?.querySelector('textarea');

    if (!form || !submitButton || !textArea) return;

    // Add custom validation before Streamlit's form submission
    form.addEventListener('submit', (event) => {
        const query = textArea.value.trim();
        if (!query) {
            event.preventDefault();
            event.stopPropagation();
            alert('Please enter a query to analyze.');
            return false;
        }

        // Show loading state
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Analyzing...';
    });
});