import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#F6F1E7",
        ivory: "#FFFDF8",
        ink: "#18211B",
        sage: "#AEBCA5",
        moss: "#466050",
        clay: "#A95E3F",
        straw: "#D9C789",
        mist: "#E8E5DC",
      },
      fontFamily: {
        display: ["var(--font-space-grotesk)", "sans-serif"],
        body: ["var(--font-dm-sans)", "sans-serif"],
        mono: ["var(--font-plex-mono)", "monospace"],
        accent: ["var(--font-newsreader)", "serif"],
      },
      boxShadow: {
        sketch: "3px 3px 0 #18211B",
        "sketch-sm": "2px 2px 0 #18211B",
      },
    },
  },
  plugins: [],
};

export default config;
