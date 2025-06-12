module.exports = {
  apps: [
    {
      name: "essaybot-express",
      script: "./src/index.ts",
      interpreter: "./node_modules/.bin/ts-node",
      instances: 1,
      autorestart: true,
      env_file: "./.env",
      env: {
        NODE_ENV: "production",
        PORT: 8000,
        FLASK_SERVICE_URL: "http://localhost:6000",
      },
    },
    {
      name: "essaybot-flask",
      script: "./src/python/app.py",
      interpreter: "./src/python/venv/bin/python",
      instances: 1,
      autorestart: true,
      env: {
        FLASK_ENV: "production",
        FLASK_PORT: 6000,
        HOST: "0.0.0.0",
      },
    },
  ],
};
