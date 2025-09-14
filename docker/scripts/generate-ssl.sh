#!/bin/bash
# SSL Certificate Generation Script for AlphaPlus Production
# This script generates self-signed certificates for development/testing
# For production, use Let's Encrypt or a trusted CA

set -e

# Configuration
DOMAIN=${DOMAIN:-"localhost"}
CERT_DIR="/etc/nginx/ssl"
DAYS=365

echo "🔐 Generating SSL certificates for AlphaPlus..."

# Create SSL directory
mkdir -p $CERT_DIR

# Generate private key
echo "📝 Generating private key..."
openssl genrsa -out $CERT_DIR/key.pem 2048

# Generate certificate signing request
echo "📝 Generating certificate signing request..."
openssl req -new -key $CERT_DIR/key.pem -out $CERT_DIR/cert.csr -subj "/C=US/ST=State/L=City/O=AlphaPlus/OU=IT/CN=$DOMAIN"

# Generate self-signed certificate
echo "📝 Generating self-signed certificate..."
openssl x509 -req -days $DAYS -in $CERT_DIR/cert.csr -signkey $CERT_DIR/key.pem -out $CERT_DIR/cert.pem

# Set proper permissions
chmod 600 $CERT_DIR/key.pem
chmod 644 $CERT_DIR/cert.pem

# Clean up CSR file
rm $CERT_DIR/cert.csr

echo "✅ SSL certificates generated successfully!"
echo "📁 Certificate location: $CERT_DIR"
echo "🔑 Private key: $CERT_DIR/key.pem"
echo "📜 Certificate: $CERT_DIR/cert.pem"
echo "⏰ Valid for: $DAYS days"

# Display certificate info
echo ""
echo "📋 Certificate Information:"
openssl x509 -in $CERT_DIR/cert.pem -text -noout | grep -E "(Subject:|Issuer:|Not Before:|Not After:)"

echo ""
echo "⚠️  IMPORTANT: This is a self-signed certificate for development/testing only."
echo "   For production, use Let's Encrypt or a trusted Certificate Authority."
echo ""
echo "🚀 To use Let's Encrypt in production, run:"
echo "   certbot --nginx -d yourdomain.com"
