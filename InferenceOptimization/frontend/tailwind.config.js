/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        primary: {
          DEFAULT: "#7C3AED",
          foreground: "#FDF7FF",
        },
        surface: {
          DEFAULT: "#1A1625",
          muted: "#232038",
          border: "#312B50",
        },
      },
    },
  },
  plugins: [],
};
