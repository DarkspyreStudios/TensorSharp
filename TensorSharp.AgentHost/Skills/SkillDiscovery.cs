// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the repository root.

using System;
using System.Collections.Generic;
using System.IO;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>Conventional repository roots shared by hosts that use local skills.</summary>
    public static class SkillDiscovery
    {
        /// <summary>
        /// Existing <c>.agents/skills</c> roots, nearest directory first, up to the
        /// nearest Git repository boundary (including worktree <c>.git</c> files).
        /// Outside a repository only the working directory itself is considered.
        /// Personal/global skill roots are never imported implicitly into a host.
        /// </summary>
        public static IReadOnlyList<string> RepositoryRoots(string workingDirectory)
        {
            if (string.IsNullOrWhiteSpace(workingDirectory) || !Directory.Exists(workingDirectory))
                return Array.Empty<string>();

            var ancestors = new List<string>();
            bool foundRepository = false;
            for (DirectoryInfo? directory = new(Path.GetFullPath(workingDirectory));
                 directory != null; directory = directory.Parent)
            {
                ancestors.Add(directory.FullName);
                string git = Path.Combine(directory.FullName, ".git");
                if (Directory.Exists(git) || File.Exists(git))
                {
                    foundRepository = true;
                    break;
                }
            }

            var roots = new List<string>();
            int count = foundRepository ? ancestors.Count : 1;
            for (int i = 0; i < count; i++)
            {
                string root = Path.Combine(ancestors[i], ".agents", "skills");
                if (Directory.Exists(root))
                    roots.Add(root);
            }
            return roots;
        }
    }
}
