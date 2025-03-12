// foldingAnimation.js - Modernized version

document.addEventListener('DOMContentLoaded', function() {
    // Initialize WOW.js for scroll animations
    new WOW({
        boxClass: 'wow',
        animateClass: 'animate__animated',
        offset: 100,
        mobile: true,
        live: true
    }).init();

    // Parallax effect for banner
    (function() {
        const parallaxElement = document.getElementById('parallax');
        
        if (parallaxElement) {
            let posY;
            
            function parallax() {
                posY = window.pageYOffset;
                parallaxElement.style.backgroundPositionY = posY * 0.5 + 'px';
            }
            
            window.addEventListener('scroll', parallax);
        }
    })();

    // Add active class to nav items based on scroll position
    (function() {
        const sections = document.querySelectorAll('section, header');
        const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
        
        function highlightNavItem() {
            let currentSection = '';
            const scrollPosition = window.pageYOffset + 100; // Add offset to account for navbar height
            
            sections.forEach(section => {
                const sectionTop = section.offsetTop;
                const sectionHeight = section.offsetHeight;
                
                if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                    currentSection = section.getAttribute('id');
                }
            });
            
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + currentSection) {
                    link.classList.add('active');
                }
            });
        }
        
        window.addEventListener('scroll', highlightNavItem);
        highlightNavItem(); // Initialize on page load
    })();
});
