-- CreateTable
CREATE TABLE "JlensShare" (
    "id" TEXT NOT NULL,
    "kind" TEXT NOT NULL,
    "modelId" TEXT NOT NULL,
    "url" TEXT NOT NULL,
    "description" TEXT,
    "lockedTokens" JSONB NOT NULL DEFAULT '[]',
    "selectedPositions" INTEGER[] DEFAULT ARRAY[]::INTEGER[],
    "activeLensModeTab" TEXT NOT NULL,
    "topN" INTEGER NOT NULL,
    "hideNonWordTokens" BOOLEAN NOT NULL DEFAULT true,
    "userId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "JlensShare_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "JlensSharePutRequest" (
    "id" TEXT NOT NULL,
    "ipAddress" TEXT NOT NULL,
    "filename" TEXT NOT NULL,
    "url" TEXT NOT NULL,
    "userId" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "JlensSharePutRequest_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "JlensShare_modelId_idx" ON "JlensShare"("modelId");

-- CreateIndex
CREATE INDEX "JlensShare_userId_idx" ON "JlensShare"("userId");

-- CreateIndex
CREATE INDEX "JlensSharePutRequest_ipAddress_idx" ON "JlensSharePutRequest"("ipAddress");

-- CreateIndex
CREATE INDEX "JlensSharePutRequest_userId_idx" ON "JlensSharePutRequest"("userId");

-- AddForeignKey
ALTER TABLE "JlensShare" ADD CONSTRAINT "JlensShare_modelId_fkey" FOREIGN KEY ("modelId") REFERENCES "Model"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "JlensShare" ADD CONSTRAINT "JlensShare_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "JlensSharePutRequest" ADD CONSTRAINT "JlensSharePutRequest_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
