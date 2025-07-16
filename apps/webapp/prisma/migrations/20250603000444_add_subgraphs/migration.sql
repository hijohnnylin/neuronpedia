-- CreateTable
CREATE TABLE "GraphMetadataSubgraph" (
    "id" TEXT NOT NULL,
    "graphMetadataId" TEXT NOT NULL,
    "pinnedIds" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "supernodes" JSONB NOT NULL DEFAULT '[]',
    "clerps" JSONB NOT NULL DEFAULT '[]',
    "pruningThreshold" DOUBLE PRECISION,
    "densityThreshold" DOUBLE PRECISION,
    "userId" TEXT,
    "isFeaturedSolution" BOOLEAN NOT NULL DEFAULT false,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "GraphMetadataSubgraph_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "GraphMetadataSubgraph" ADD CONSTRAINT "GraphMetadataSubgraph_graphMetadataId_fkey" FOREIGN KEY ("graphMetadataId") REFERENCES "GraphMetadata"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "GraphMetadataSubgraph" ADD CONSTRAINT "GraphMetadataSubgraph_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;
