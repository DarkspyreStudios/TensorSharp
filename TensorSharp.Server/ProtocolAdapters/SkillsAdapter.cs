// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.ProtocolAdapters;

/// <summary>
/// The ASP.NET Core transport for the Agent Skills management surface. Every route
/// is a thin shell over <see cref="SkillsService"/> in TensorSharp.Chat, which owns
/// the payload shapes and the refusals (thrown as
/// <see cref="WebUiRequestRejectedException"/> and answered here as status + JSON);
/// what remains here is what only HTTP has — the multipart form, the
/// <c>text/plain</c> file response and the <c>201</c> on install.
/// </summary>
public sealed class SkillsAdapter
{
    private readonly SkillsService _service;

    public SkillsAdapter(
        SkillRegistry skills,
        ServerHostingOptions options,
        UploadStoragePolicy uploads,
        ILoggerFactory loggerFactory)
    {
        _service = new SkillsService(skills, options, uploads, loggerFactory);
    }

    private static IResult Rejected(WebUiRequestRejectedException ex) =>
        Results.Json(ex.Payload, statusCode: ex.StatusCode);

    // ---- /v1 ---------------------------------------------------------------

    /// <summary><c>GET /v1/skills</c> — the OpenAI-shaped list.</summary>
    public IResult ListV1() => Results.Json(_service.ListV1());

    /// <summary><c>GET /v1/skills/{name}</c> — one skill, including its instructions.</summary>
    public IResult GetV1(string name)
    {
        try
        {
            return Results.Json(_service.GetV1(name));
        }
        catch (WebUiRequestRejectedException ex)
        {
            return Rejected(ex);
        }
    }

    // ---- /api --------------------------------------------------------------

    /// <summary><c>GET /api/skills</c> — the Web UI's roster, capabilities and load errors.</summary>
    public IResult ListForUi() => Results.Json(_service.ListForUi());

    /// <summary><c>GET /api/skills/{name}</c> — one skill with its instructions.</summary>
    public IResult GetForUi(string name)
    {
        try
        {
            return Results.Json(_service.GetForUi(name));
        }
        catch (WebUiRequestRejectedException ex)
        {
            return Rejected(ex);
        }
    }

    /// <summary>
    /// <c>GET /api/skills/{name}/files/{*path}</c> — one bundled file, always as
    /// <c>text/plain</c>: a skill may ship an <c>.html</c> or <c>.js</c> file, and serving
    /// it with its real type would execute uploaded content in the server's own origin.
    /// </summary>
    public IResult GetFile(string name, string path)
    {
        if (!_service.TryGetFile(name, path, out string text, out object error))
            return Results.Json(error, statusCode: StatusCodes.Status404NotFound);
        return Results.Text(text, "text/plain; charset=utf-8");
    }

    /// <summary>
    /// <c>POST /api/skills</c> — install a skill from a ZIP in a multipart form
    /// (optional field <c>overwrite=true|1</c>). The 403 for a host that takes no
    /// uploads is answered before the form is read.
    /// </summary>
    public async Task<IResult> InstallAsync(HttpRequest request)
    {
        try
        {
            _service.EnsureInstallable();

            if (!request.HasFormContentType)
                return Results.Json(new { error = "Expected a multipart/form-data upload." }, statusCode: StatusCodes.Status400BadRequest);

            IFormCollection form = await request.ReadFormAsync().ConfigureAwait(false);
            IFormFile? file = form.Files.Count == 0 ? null : form.Files[0];
            bool overwrite = ReadBool(form, "overwrite");

            // A missing part reaches the service as a null stream, which it refuses
            // with the same 400 an empty file gets.
            Stream stream = file?.OpenReadStream() ?? Stream.Null;
            await using (stream)
            {
                object installed = _service.Install(stream, file?.FileName, file?.Length ?? 0, overwrite);
                return Results.Json(installed, statusCode: StatusCodes.Status201Created);
            }
        }
        catch (WebUiRequestRejectedException ex)
        {
            return Rejected(ex);
        }
    }

    /// <summary><c>DELETE /api/skills/{name}</c> — remove an installed skill.</summary>
    public IResult Remove(string name)
    {
        try
        {
            return Results.Json(_service.Remove(name));
        }
        catch (WebUiRequestRejectedException ex)
        {
            return Rejected(ex);
        }
    }

    /// <summary><c>POST /api/skills/rescan</c> — pick up changes made on disk without a restart.</summary>
    public IResult Rescan() => Results.Json(_service.Rescan());

    /// <summary>The capability block folded into <c>GET /api/models</c>.</summary>
    public object DescribeCapabilities() => _service.DescribeCapabilities();

    private static bool ReadBool(IFormCollection form, string key) =>
        form.TryGetValue(key, out Microsoft.Extensions.Primitives.StringValues values)
        && values.Count > 0
        && (string.Equals(values[0], "true", StringComparison.OrdinalIgnoreCase)
            || string.Equals(values[0], "1", StringComparison.Ordinal));
}
