-- CreateEnum
CREATE TYPE "ProblemNodeType" AS ENUM ('topic', 'paper', 'tool', 'dataset', 'eval', 'replication');

-- CreateEnum
CREATE TYPE "ProblemNodeApprovalState" AS ENUM ('APPROVED', 'REJECTED', 'PENDING');

-- CreateEnum
CREATE TYPE "ProblemEdgeType" AS ENUM ('raises', 'addresses', 'extends', 'replicates', 'related_to');

-- AlterTable
ALTER TABLE "User" ADD COLUMN     "isProblemEditor" BOOLEAN NOT NULL DEFAULT false;

-- CreateTable
CREATE TABLE "ProblemNode" (
    "id" TEXT NOT NULL,
    "type" "ProblemNodeType" NOT NULL,
    "parentId" TEXT,
    "title" TEXT,
    "description" TEXT,
    "mainUrl" TEXT,
    "additionalUrls" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "applicationTags" TEXT[] DEFAULT ARRAY[]::TEXT[],
    "approvalState" "ProblemNodeApprovalState" NOT NULL DEFAULT 'PENDING',
    "approverId" TEXT,
    "createdById" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ProblemNode_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProblemEdge" (
    "id" TEXT NOT NULL,
    "sourceNodeId" TEXT NOT NULL,
    "targetNodeId" TEXT NOT NULL,
    "type" "ProblemEdgeType" NOT NULL,
    "createdById" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "ProblemEdge_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProblemNodeComment" (
    "id" TEXT NOT NULL,
    "problemNodeId" TEXT NOT NULL,
    "parentCommentId" TEXT,
    "text" TEXT NOT NULL,
    "userId" TEXT NOT NULL,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updatedAt" TIMESTAMP(3) NOT NULL,

    CONSTRAINT "ProblemNodeComment_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "ProblemNodeLog" (
    "id" TEXT NOT NULL,
    "timestamp" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "userId" TEXT NOT NULL,
    "problemNodeId" TEXT NOT NULL,
    "action" TEXT NOT NULL,
    "details" TEXT,

    CONSTRAINT "ProblemNodeLog_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "ProblemNode_parentId_idx" ON "ProblemNode"("parentId");

-- CreateIndex
CREATE INDEX "ProblemNode_type_idx" ON "ProblemNode"("type");

-- CreateIndex
CREATE INDEX "ProblemNode_approvalState_idx" ON "ProblemNode"("approvalState");

-- CreateIndex
CREATE INDEX "ProblemNode_createdById_idx" ON "ProblemNode"("createdById");

-- CreateIndex
CREATE INDEX "ProblemEdge_sourceNodeId_idx" ON "ProblemEdge"("sourceNodeId");

-- CreateIndex
CREATE INDEX "ProblemEdge_targetNodeId_idx" ON "ProblemEdge"("targetNodeId");

-- CreateIndex
CREATE UNIQUE INDEX "ProblemEdge_sourceNodeId_targetNodeId_type_key" ON "ProblemEdge"("sourceNodeId", "targetNodeId", "type");

-- CreateIndex
CREATE INDEX "ProblemNodeComment_problemNodeId_idx" ON "ProblemNodeComment"("problemNodeId");

-- CreateIndex
CREATE INDEX "ProblemNodeComment_parentCommentId_idx" ON "ProblemNodeComment"("parentCommentId");

-- CreateIndex
CREATE INDEX "ProblemNodeComment_userId_idx" ON "ProblemNodeComment"("userId");

-- CreateIndex
CREATE INDEX "ProblemNodeLog_problemNodeId_idx" ON "ProblemNodeLog"("problemNodeId");

-- CreateIndex
CREATE INDEX "ProblemNodeLog_userId_idx" ON "ProblemNodeLog"("userId");

-- CreateIndex
CREATE INDEX "ProblemNodeLog_timestamp_idx" ON "ProblemNodeLog"("timestamp");

-- AddForeignKey
ALTER TABLE "ProblemNode" ADD CONSTRAINT "ProblemNode_parentId_fkey" FOREIGN KEY ("parentId") REFERENCES "ProblemNode"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemNode" ADD CONSTRAINT "ProblemNode_approverId_fkey" FOREIGN KEY ("approverId") REFERENCES "User"("id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemNode" ADD CONSTRAINT "ProblemNode_createdById_fkey" FOREIGN KEY ("createdById") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemEdge" ADD CONSTRAINT "ProblemEdge_sourceNodeId_fkey" FOREIGN KEY ("sourceNodeId") REFERENCES "ProblemNode"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemEdge" ADD CONSTRAINT "ProblemEdge_targetNodeId_fkey" FOREIGN KEY ("targetNodeId") REFERENCES "ProblemNode"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemEdge" ADD CONSTRAINT "ProblemEdge_createdById_fkey" FOREIGN KEY ("createdById") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemNodeComment" ADD CONSTRAINT "ProblemNodeComment_problemNodeId_fkey" FOREIGN KEY ("problemNodeId") REFERENCES "ProblemNode"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemNodeComment" ADD CONSTRAINT "ProblemNodeComment_parentCommentId_fkey" FOREIGN KEY ("parentCommentId") REFERENCES "ProblemNodeComment"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemNodeComment" ADD CONSTRAINT "ProblemNodeComment_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemNodeLog" ADD CONSTRAINT "ProblemNodeLog_userId_fkey" FOREIGN KEY ("userId") REFERENCES "User"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "ProblemNodeLog" ADD CONSTRAINT "ProblemNodeLog_problemNodeId_fkey" FOREIGN KEY ("problemNodeId") REFERENCES "ProblemNode"("id") ON DELETE CASCADE ON UPDATE CASCADE;
