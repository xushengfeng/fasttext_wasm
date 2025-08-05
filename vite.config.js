import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
	// 打包配置
	build: {
		lib: {
			entry: resolve(__dirname, "src/fasttext.ts"),
			name: "fasttext",
			fileName: (format) => `fasttext.${format}.js`,
		},
	},
});
