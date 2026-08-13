import antfu from '@antfu/eslint-config';

export default antfu({
  ignores: [
    '.codegraph/**',
    '.vite/**',
    'apps/desktop/.vite/**',
    'apps/desktop/out/**',
    'coverage/**',
    'dist/**',
    'node_modules/**',
    '**/*.md',
    'pnpm-lock.yaml',
  ],
  formatters: true,
  jsonc: true,
  markdown: false,
  pnpm: true,
  stylistic: {
    braceStyle: '1tbs',
    indent: 2,
    quotes: 'single',
    semi: true,
  },
  typescript: true,
  vue: true,
  yaml: true,
}, {
  files: ['apps/desktop/renderer/src/**/*.vue'],
  rules: {
    'vue/block-order': ['error', { order: ['script', 'template', 'style'] }],
    'vue/component-name-in-template-casing': ['error', 'PascalCase', {
      registeredComponentsOnly: false,
    }],
    'vue/html-self-closing': 'off',
  },
}, {
  files: ['apps/desktop/main/**/*.ts', 'apps/desktop/preload/**/*.ts', 'apps/desktop/scripts/**/*.ts', 'apps/desktop/*.ts'],
  rules: {
    'node/prefer-global/process': 'off',
  },
}, {
  files: ['packages/**/*.test.ts', 'scripts/**/*.test.mjs', 'services/**/*.test.ts'],
  rules: {
    'test/no-import-node-test': 'off',
  },
});
