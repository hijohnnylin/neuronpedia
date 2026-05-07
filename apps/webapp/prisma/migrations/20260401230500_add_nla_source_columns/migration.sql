-- AlterTable
ALTER TABLE "NlaSource" ADD COLUMN "name" TEXT NOT NULL DEFAULT '',
ADD COLUMN "description" TEXT NOT NULL DEFAULT '',
ADD COLUMN "url" TEXT NOT NULL DEFAULT '',
ADD COLUMN "author" TEXT NOT NULL DEFAULT '';

-- CreateIndex
CREATE UNIQUE INDEX "NlaSource_name_key" ON "NlaSource"("name");
