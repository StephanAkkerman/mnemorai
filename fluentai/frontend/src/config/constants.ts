const isGithubPages = process.env.NODE_ENV === "production" && process.env.GITHUB_PAGES === "true";

export const ANKI_CONFIG = {
  API_URL: isGithubPages ? 'http://127.0.0.1:8765' : '/api/anki',
  VERSION: 6,
  DEFAULT_DECK: 'Model Deck',
  DEFAULT_MODEL: 'Basic',
  DEFAULT_TAGS: ['FluentAI']
} as const;

