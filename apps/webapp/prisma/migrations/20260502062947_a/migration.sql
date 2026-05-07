-- AlterTable
ALTER TABLE "NlaExplainCache" ADD COLUMN     "textHash" VARCHAR(64);

-- CreateTable
CREATE TABLE "NlaExplainShare" (
    "id" TEXT NOT NULL,
    "cacheId" TEXT NOT NULL,
    "position" INTEGER,
    "paragraph" INTEGER,
    "highlightStart" INTEGER,
    "highlightEnd" INTEGER,
    "comment" TEXT NOT NULL DEFAULT '',
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "NlaExplainShare_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "NlaExplainShare_cacheId_idx" ON "NlaExplainShare"("cacheId");

-- AddForeignKey
ALTER TABLE "NlaExplainShare" ADD CONSTRAINT "NlaExplainShare_cacheId_fkey" FOREIGN KEY ("cacheId") REFERENCES "NlaExplainCache"("id") ON DELETE CASCADE ON UPDATE CASCADE;
