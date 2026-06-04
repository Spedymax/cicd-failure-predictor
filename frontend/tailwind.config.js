/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        risk: {
          low: "#22c55e",
          warn: "#f59e0b",
          high: "#ef4444",
        },
      },
    },
  },
  plugins: [],
};
