const { defineConfig } = require('@vue/cli-service')
module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8002',
        changeOrigin: true,
        pathRewrite: { '^/api': '' },
        ws: true,
      },
      '/ws': {
        target: 'http://127.0.0.1:8002',
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
