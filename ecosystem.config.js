module.exports = {
  apps: [
    {
      name: "essaybot-express",
      script: "./src/index.ts",
      interpreter: "npx",
      interpreter_args: "ts-node",
      instances: 1,
      autorestart: true,
      cwd: "/opt/LearnBot/EssayBot-Server",
      env_file: "/opt/LearnBot/EssayBot-Server/.env",
      output: "/var/log/pm2/essaybot-express-out.log",
      error: "/var/log/pm2/essaybot-express-error.log",
      log: "/var/log/pm2/essaybot-express-combined.log",
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
      cwd: "/opt/LearnBot/EssayBot-Server",
      output: "/var/log/pm2/essaybot-flask-out.log",
      error: "/var/log/pm2/essaybot-flask-error.log",
      env: {
        FLASK_ENV: "production",
        FLASK_PORT: 6000,
        HOST: "0.0.0.0",
      },
    },
  ],
};
