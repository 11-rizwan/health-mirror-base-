document.addEventListener('DOMContentLoaded', () => {
    const navToggle = document.getElementById('nav-toggle');
    const navMenuWrapper = document.getElementById('nav-menu-wrapper');

    if (navToggle && navMenuWrapper) {
        navToggle.addEventListener('click', () => {
            // Toggle the 'is-active' class on both the icon and the menu
            navToggle.classList.toggle('is-active');
            navMenuWrapper.classList.toggle('is-active');
        });
    }
});