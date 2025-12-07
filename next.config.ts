import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,

  // Keep TensorFlow.js server-side only (Next.js 16+ syntax)
  serverExternalPackages: ['@tensorflow/tfjs', '@tensorflow/tfjs-node'],

  // Optimize barrel imports for faster compilation
  experimental: {
    optimizePackageImports: [
      '@/components/ui',
      '@/components/stocks',
      '@/components/layout',
    ],
  },

  // Turbopack config
  turbopack: {},
};

export default nextConfig;
