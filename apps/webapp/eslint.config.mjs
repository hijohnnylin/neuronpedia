import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";
import tseslint from "typescript-eslint";
import prettier from "eslint-config-prettier";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({ baseDirectory: __dirname });

export default [
  ...compat.extends("next/typescript"),
  { ignores: ["prisma/generated/zod/*", ".next/*", "coverage/*", "types/eval_output_schema_*.d.ts", "next-env.d.ts"] },
  ...compat.extends("next/core-web-vitals"),
  ...tseslint.configs.recommended,
  prettier,
  {
    rules: {
      "@next/next/no-html-link-for-pages": "off",
      "no-nested-ternary": "off",
      "no-use-before-define": "off",
      "no-alert": "warn",
      "no-restricted-syntax": "warn",
      "no-await-in-loop": "warn",
      "react/jsx-filename-extension": [2, { extensions: [".js", ".jsx", ".ts", ".tsx"] }],
      "max-len": "off",

      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          args: "after-used",
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrors: "none",
          destructuredArrayIgnorePattern: "^_",
          ignoreRestSiblings: true,
        },
      ],
      "@typescript-eslint/no-unused-expressions": [
        "error",
        { allowShortCircuit: true, allowTernary: true, allowTaggedTemplates: true },
      ],
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/ban-ts-comment": "warn",
      "@typescript-eslint/no-require-imports": "off",
    },
  }
];
