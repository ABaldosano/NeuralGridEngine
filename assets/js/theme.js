"use strict";

(function initTheme() {
  const STORAGE_KEY = "neuralgrid-theme";
  const root = document.documentElement;
  const saved = localStorage.getItem(STORAGE_KEY);
  // Light is the product's default look regardless of the device's system
  // color scheme. Only an explicit in-app toggle (saved below) should
  // switch a visitor to dark — we don't infer it from OS preference.
  root.setAttribute("data-theme", saved || "light");

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
    const closeBtn = document.getElementById("mobileMenuClose");
    if (!hamburger || !mobileMenu || !overlay) return;

    const closeMenu = () => {
      hamburger.classList.remove("open");
      mobileMenu.classList.remove("open");
      overlay.classList.remove("open");
      hamburger.setAttribute("aria-expanded", "false");
      document.body.style.overflow = "";
    };

    const openMenu = () => {
      hamburger.classList.add("open");
      mobileMenu.classList.add("open");
      overlay.classList.add("open");
      hamburger.setAttribute("aria-expanded", "true");
      document.body.style.overflow = "hidden";
    };

    hamburger.addEventListener("click", () => {
      if (mobileMenu.classList.contains("open")) closeMenu();
      else openMenu();
    });

    overlay.addEventListener("click", closeMenu);
    if (closeBtn) closeBtn.addEventListener("click", closeMenu);

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