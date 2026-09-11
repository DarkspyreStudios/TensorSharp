// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;

namespace InferenceWeb.Tests;

public class DeepSeek41ObservedToolMarkupTests
{
    // Verbatim tool blocks from the Q2_K HTTP run on 2026-09-11. The first
    // previously dispatched get_weather({}); the second dropped the call.
    private const string PlainWeather = """
        <｜DSML｜ calls>
        <｜DSML｜ invoke name="get_weather">
            <parameter name="city">Paris</parameter>
            <parameter name="units">celsius</parameter>
        </｜DSML｜ invoke>
        </｜DSML｜ calls>
        """;
    private const string MixedWeather = """
        <｜DSML｜ calls>
        <｜DSML｜ invoke name="get_weather">
        <｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>
        <｜DSML｜ parameter name="units" string="true">celsius</parameter>
        </｜DSML｜ invoke>
        </｜DSML｜ calls>
        """;

    [Theory]
    [InlineData(PlainWeather)]
    [InlineData(MixedWeather)]
    public void ObservedWeatherCallsSurviveEveryStreamBoundary(string input)
    {
        for (int split = 0; split <= input.Length; split++)
        {
            var parser = new DeepSeek41OutputParser();
            parser.Init(false, []);
            var first = parser.Add(input[..split], false);
            var last = parser.Add(input[split..], true);
            Assert.Empty(first.Content + last.Content);
            var call = Assert.Single((first.ToolCalls ?? []).Concat(last.ToolCalls ?? []));
            Assert.Equal("get_weather", call.Name);
            Assert.Equal("Paris", call.Arguments["city"]);
            Assert.Equal("celsius", call.Arguments["units"]);
            Assert.Equal(2, call.Arguments.Count);
        }
    }

    [Fact]
    public void ObservedInvoiceCallSurvivesCharacterStreaming()
    {
        const string input = "I'll start by reading invoice INV-472 to get its unit price and quantity.\n\n" +
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read_invoice\">\n" +
            "<parameter name=\"invoice_id\">INV-472</parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>";
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, []);
        var content = new StringBuilder();
        var calls = new List<ToolCall>();
        foreach (char character in input)
        {
            var part = parser.Add(character.ToString(), false);
            content.Append(part.Content);
            calls.AddRange(part.ToolCalls ?? []);
        }
        var last = parser.Add("", true);
        content.Append(last.Content);
        calls.AddRange(last.ToolCalls ?? []);
        Assert.Equal("I'll start by reading invoice INV-472 to get its unit price and quantity.\n\n", content.ToString());
        var call = Assert.Single(calls);
        Assert.Equal("read_invoice", call.Name);
        Assert.Equal("INV-472", call.Arguments["invoice_id"]);
    }

    [Theory]
    [InlineData("<parameter name=\"code\">  café\n<b>東京</b> & 😀\n</parameter>")]
    [InlineData("<｜DSML｜ parameter name=\"code\" string=\"true\">  café\n<b>東京</b> & 😀\n</parameter>")]
    public void AlternateTagsPreserveStringWhitespaceAndLiteralMarkup(string parameters)
    {
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, []);
        var output = parser.Add(Wrap(parameters), true);
        Assert.Equal("  café\n<b>東京</b> & 😀\n", Assert.Single(output.ToolCalls!).Arguments["code"]);
    }

    [Theory]
    [InlineData("<argument name=\"city\">Paris</argument>")]
    [InlineData("<parameter name=\"city\">Paris")]
    [InlineData("<parameter name=\"city\">Paris</wrong>")]
    [InlineData("<parameter name=\"city\">Paris<parameter name=\"units\">celsius</parameter>")]
    [InlineData("<parameter name=\"city\">Paris</parameter><parameter name=\"units\">celsius")]
    [InlineData("<parameter name=\"city\">Paris</parameter>unparsed argument")]
    [InlineData("<parameter name=\"city\">Paris</parameter><parameter name=\"city\">Berlin</parameter>")]
    [InlineData("<parameter name=\"\">Paris</parameter>")]
    [InlineData("<parameter name=\"city\" string=\"sometimes\">Paris</parameter>")]
    [InlineData("<parameter name=\"count\" string=\"false\">{</parameter>")]
    [InlineData("<parameter name=\"count\" string=\"false\"></parameter>")]
    public void MalformedMarkupNeverDispatchesEmptyOrPartialArguments(string parameters)
    {
        string input = Wrap(parameters);
        for (int split = 0; split <= input.Length; split++)
        {
            var parser = new DeepSeek41OutputParser();
            parser.Init(false, []);
            Assert.Null(parser.Add(input[..split], false).ToolCalls);
            Assert.Null(parser.Add(input[split..], true).ToolCalls);
        }
    }

    [Fact]
    public void PlainJsonParameterPreservesTypeAndRequiresCompleteInvoke()
    {
        const string parameters = "<parameter name=\"count\" string=\"false\">13.75</parameter>";
        var parser = new DeepSeek41OutputParser();
        parser.Init(false, []);
        var output = parser.Add(Wrap(parameters), true);
        Assert.Equal(13.75, Assert.Single(output.ToolCalls!).Arguments["count"]);
        parser.Init(false, []);
        Assert.Null(parser.Add("<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">" + parameters, true).ToolCalls);
    }

    private static string Wrap(string parameters)
        => "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">" + parameters + "</｜DSML｜ invoke></｜DSML｜ calls>";
}
