-- DropIndex
DROP INDEX "NlaSource_name_key";

-- CreateIndex
CREATE UNIQUE INDEX "NlaSource_modelId_name_key" ON "NlaSource"("modelId", "name");
