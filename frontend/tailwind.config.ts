import type { Config } from "tailwindcss";

export default {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
      },
      boxShadow: {
        'teal-glow-xs': '0 0 8px 8px rgba(56,191,248,0.9), 0 0 8px 8px rgba(45,212,191,0.9)',
        'teal-glow-sm': '0 0 20px 5px rgba(56,191,248,0.9), 0 0 20px 5px rgba(45,212,191,0.9)',
        'teal-glow-md': '0 0 40px 10px rgba(56,191,248,0.9), 0 0 40px 10px rgba(45,212,191,0.9)',
      },
    },
  },
  variants: {
    extend: {
      transform: ["hover", "focus"],
      translate: ["motion-safe"],
      rotate: ["hover", "focus"],
    },
  },
  plugins: [],
} satisfies Config;
