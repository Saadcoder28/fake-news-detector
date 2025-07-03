FROM nginx:1.27-alpine

# Copy the frontend build into nginx's web root
COPY web/dist /usr/share/nginx/html

# Replace default nginx config with your own
COPY proxy/nginx.conf /etc/nginx/nginx.conf

# Remove default config to avoid conflicts
RUN rm /etc/nginx/conf.d/default.conf
