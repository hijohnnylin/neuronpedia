-- CreateIndex
CREATE INDEX "NlaExplainCache_modelId_idx" ON "NlaExplainCache"("modelId");

-- CreateIndex
CREATE INDEX "NlaExplainShare_featured_idx" ON "NlaExplainShare"("featured");

-- AddForeignKey
ALTER TABLE "NlaExplainCache" ADD CONSTRAINT "NlaExplainCache_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;
