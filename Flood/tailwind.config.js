/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,jsx,ts,tsx}",
    "./components/**/*.{js,jsx,ts,tsx}"
  ],
  presets: [require("nativewind/preset")],
  theme: {
    extend: {
      colors: {
        primary: '#0ea5e9',
        secondary: '#16a34a',
        accent: '#8b5cf6',
        danger: '#ef4444',
      },
    },
  },
  plugins: [],
}