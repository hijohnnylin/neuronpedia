import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Steerify",
  description: "A simple interface for steering language models using features from Neuronpedia",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <main className="min-h-screen p-8">
          <header className="mb-8">
            <h1 className="text-3xl font-bold">Steerify</h1>
            <p className="text-muted-foreground">
              Search for features and steer language models
            </p>
          </header>
          {children}
        </main>
      </body>
    </html>
  );
}