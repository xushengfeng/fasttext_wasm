import { defineConfig } from "vite";
import { resolve } from "node:path";
import Binary from "vite-plugin-binary";

export default defineConfig({
    // 打包配置
    build: {
        lib: {
            entry: resolve(__dirname, "src/fasttext.ts"),
            name: "fasttext",
            fileName: (format) => `fasttext.${format}.js`,
        },
    },
    plugins: [Binary({ gzip: false })],
});
