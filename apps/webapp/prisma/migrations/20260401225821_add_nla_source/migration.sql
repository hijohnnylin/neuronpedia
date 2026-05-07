-- CreateTable
CREATE TABLE "NlaSource" (
    "id" TEXT NOT NULL,
    "modelId" TEXT NOT NULL,
    "actor" TEXT NOT NULL,
    "critic" TEXT NOT NULL,
    "servers" TEXT[],
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "NlaSource_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "NlaSource_modelId_idx" ON "NlaSource"("modelId");

-- CreateIndex
CREATE UNIQUE INDEX "NlaSource_modelId_actor_critic_key" ON "NlaSource"("modelId", "actor", "critic");

-- AddForeignKey
ALTER TABLE "NlaSource" ADD CONSTRAINT "NlaSource_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;
