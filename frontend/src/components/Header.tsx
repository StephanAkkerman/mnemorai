"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useState } from "react";
import lightLogo from "../../public/Main Logo (light).svg";
import darkLogo from "../../public/Main Logo (dark).svg";
import { Sun, Moon, Menu, X } from "lucide-react";

export default function Header() {
  const [darkMode, setDarkMode] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    // Check system preference initially
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setDarkMode(prefersDark);

    if (prefersDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    if (darkMode) {
      document.documentElement.classList.remove("dark");
    } else {
      document.documentElement.classList.add("dark");
    }
  };

  return (
    <header className="sticky top-0 z-50 w-full bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 shadow-sm transition-all duration-300">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <Link href="/" className="flex items-center space-x-3">
              <div className="relative w-10 h-10 transition-transform duration-300 hover:rotate-12">
                <Image
                  src={darkMode ? darkLogo : lightLogo}
                  alt="mnemorai Logo"
                  className="object-contain"
                  fill
                  sizes="2.5rem"
                  priority
                />
              </div>
              <h1 className="text-2xl font-bold gradient-text">
                mnemorai
              </h1>
            </Link>
          </div>

          {/* Desktop navigation */}
          <nav className="hidden md:flex items-center space-x-8">
            <Link
              href="/"
              className="text-gray-600 dark:text-gray-300 hover:text-blue-500 dark:hover:text-blue-400 font-medium transition-colors"
            >
              Home
            </Link>
            <Link href="/card-gen" className="text-gray-600 dark:text-gray-300 hover:text-blue-500 dark:hover:text-blue-400 font-medium transition-colors">
              Card Generation
            </Link>
            <Link
              href="/library"
              className="text-gray-600 dark:text-gray-300 hover:text-blue-500 dark:hover:text-blue-400 font-medium transition-colors"
            >
              Library
            </Link>
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 
                         hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors duration-300"
              aria-label="Toggle Dark Mode"
            >
              {darkMode ? <Sun size={20} /> : <Moon size={20} />}
            </button>
          </nav>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 
                         hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors duration-300 mr-2"
              aria-label="Toggle Dark Mode"
            >
              {darkMode ? <Sun size={18} /> : <Moon size={18} />}
            </button>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-md text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
            >
              {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile menu */}
      <div className={`md:hidden transition-all duration-300 ease-in-out ${mobileMenuOpen ? 'max-h-36 opacity-100' : 'max-h-0 opacity-0 overflow-hidden'}`}>
        <div className="px-4 pt-2 pb-4 space-y-2 bg-white dark:bg-gray-900 shadow-lg">
          <Link
            href="/"
            className="block py-2 text-gray-600 dark:text-gray-300 hover:text-blue-500 dark:hover:text-blue-400 font-medium"
            onClick={() => setMobileMenuOpen(false)}
          >
            Home
          </Link>
          <Link
            href="/card-gen"
            className="block py-2 text-gray-600 dark:text-gray-300 hover:text-blue-500 dark:hover:text-blue-400 font-medium"
            onClick={() => setMobileMenuOpen(false)}
          >
            Card Generation
          </Link>
          <Link
            href="/library"
            className="block py-2 text-gray-600 dark:text-gray-300 hover:text-blue-500 dark:hover:text-blue-400 font-medium"
            onClick={() => setMobileMenuOpen(false)}
          >
            Library
          </Link>
        </div>
      </div>
    </header>
  );
}
