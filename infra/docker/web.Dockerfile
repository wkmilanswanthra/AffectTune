FROM node:20-alpine
WORKDIR /app
COPY apps/web/package.json apps/web/pnpm-lock.yaml ./
RUN corepack enable && pnpm i --frozen-lockfile
COPY apps/web .
EXPOSE 3000
CMD ["pnpm","dev","-p","3000","--host"]
