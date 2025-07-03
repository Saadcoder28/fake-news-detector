# proxy/Dockerfile  (minimal, final version)
FROM nginx:1.27-alpine

# static site (built by Vite)
COPY dist /usr/share/nginx/html

# replace default config with ours
COPY nginx.conf /etc/nginx/nginx.conf
RUN rm /etc/nginx/conf.d/default.conf   # avoid double server{}

EXPOSE 80
HEALTHCHECK CMD wget -qO- http://localhost/health || exit 1
