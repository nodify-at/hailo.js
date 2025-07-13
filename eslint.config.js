import js from '@eslint/js'
import globals from 'globals'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'
import eslintConfigPrettier from 'eslint-config-prettier/flat'

export default defineConfig([
    globalIgnores(['dist/*', 'eslint.config.js']),
    { files: ['**/*.{js,mjs,cjs,ts}'], plugins: { js }, extends: ['js/recommended'] },
    { files: ['**/*.{js,mjs,cjs,ts}'], languageOptions: { globals: globals.node }, ignores: ['./dist/**/*.js'] },
    tseslint.configs.recommended,
    {
        plugins: { '@typescript-eslint': tseslint.plugin },
        languageOptions: { parser: tseslint.parser, parserOptions: { project: './tsconfig.json' } },
        rules: {
            '@typescript-eslint/no-floating-promises': 'error',
            '@typescript-eslint/explicit-function-return-type': 'error',
            '@typescript-eslint/explicit-member-accessibility': [2, { accessibility: 'no-public' }],
        },
    },
    eslintConfigPrettier,
])
