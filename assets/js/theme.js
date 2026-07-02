"use strict";

(function initTheme() {
  const STORAGE_KEY = "neuralgrid-theme";
  const root = document.documentElement;
  const saved = localStorage.getItem(STORAGE_KEY);
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.setAttribute("data-theme", saved || (prefersDark ? "dark" : "light"));

  document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.getElementById("themeToggle");
    if (toggle) {
      toggle.addEventListener("click", () => {
        const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        localStorage.setItem(STORAGE_KEY, next);
      });
    }

    const hamburger = document.getElementById("hamburgerBtn");
    const mobileMenu = document.getElementById("mobileMenu");
    const overlay = document.getElementById("mobileMenuOverlay");
    if (!hamburger || !mobileMenu || !overlay) return;

    const closeMenu = () => {
      hamburger.classList.remove("open");
      hamburger.setAttribute("aria-expanded", "false");
      mobileMenu.hidden = true;
      overlay.hidden = true;
      document.body.classList.remove("menu-open");
    };

    const openMenu = () => {
      hamburger.classList.add("open");
      hamburger.setAttribute("aria-expanded", "true");
      mobileMenu.hidden = false;
      overlay.hidden = false;
      document.body.classList.add("menu-open");
    };

    hamburger.addEventListener("click", () => {
      if (mobileMenu.hidden) openMenu();
      else closeMenu();
    });

    overlay.addEventListener("click", closeMenu);

    mobileMenu.querySelectorAll("a").forEach((link) => {
      link.addEventListener("click", closeMenu);
    });

    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeMenu();
    });

    window.addEventListener("resize", () => {
      if (window.innerWidth > 720) closeMenu();
    });
  });
})();