fn render_help(frame: &mut Frame<'_>, app: &App) {
    let area = centered_rect(frame.area(), 66, 72, 52, 18);
    frame.render_widget(Clear, area);
    let mut lines = match app.help_context() {
        HelpContext::ModelSelector => vec![
            Line::styled("MODEL SELECTION", Style::default().fg(SIGNAL).bold()),
            Line::raw("Type/paste     edit source path"),
            Line::raw("←/→ Home/End move path cursor"),
            Line::raw("Shift+←/→     select path text"),
            Line::raw("Tab            path / browser / action"),
            Line::raw("↑/↓             browse or scroll diagnostics"),
            Line::raw("Enter          open directory or run action"),
            Line::raw("Esc            close model selection"),
        ],
        HelpContext::Compiler => vec![
            Line::styled("MODEL COMPILER", Style::default().fg(SIGNAL).bold()),
            Line::raw("↑/↓             scroll diagnostics"),
            Line::raw("Esc            request safe cancellation"),
            Line::raw("Compilation continues while Help is open."),
        ],
        HelpContext::Node => vec![
            Line::styled("NODE INSTANCE", Style::default().fg(SIGNAL).bold()),
            Line::raw("↑/↓ or Tab      move modal focus"),
            Line::raw("←/→             change selected control"),
            Line::raw("Type/paste     edit a text or numeric value"),
            Line::raw("Enter          toggle, apply, or cancel"),
            Line::raw("A              expand/collapse anatomy"),
            Line::raw("PgUp/PgDn      scroll expanded anatomy"),
            Line::raw("Esc            discard uncommitted edits"),
        ],
        HelpContext::Sequence => vec![
            Line::styled("LAYER SEQUENCE", Style::default().fg(SIGNAL).bold()),
            Line::raw("Type/paste     edit the numeric sequence"),
            Line::raw("←/→ Home/End move text cursor"),
            Line::raw("Shift+←/→     select text"),
            Line::raw("Ctrl+A         select all"),
            Line::raw("Tab / Esc      return to execution graph"),
        ],
        HelpContext::Graph => vec![
            Line::styled("EXECUTION GRAPH", Style::default().fg(SIGNAL).bold()),
            Line::raw("←/→ or h/l    select node"),
            Line::raw("Enter          edit selected instance"),
            Line::raw("Ctrl+D         duplicate selected instance"),
            Line::raw("Delete         remove selected instance"),
            Line::raw("Alt+←/→        move selected instance"),
            Line::raw("Scroll         pan the graph"),
            Line::raw("Tab            edit layer sequence"),
            Line::raw("R              toggle residency policy"),
            Line::raw("Ctrl+R         refresh runtime devices"),
        ],
    };
    lines.extend([
        Line::raw(""),
        Line::styled("GLOBAL", Style::default().fg(SIGNAL).bold()),
        Line::raw("F1             close this help"),
        Line::raw("Ctrl+M         enable/disable mouse capture"),
        Line::raw("Ctrl+C         quit or safely cancel compiler"),
    ]);
    frame.render_widget(
        Paragraph::new(lines)
            .block(
                Block::default()
                    .title(" HELP ")
                    .borders(Borders::ALL)
                    .border_type(BorderType::Double)
                    .border_style(Style::default().fg(SIGNAL)),
            )
            .wrap(Wrap { trim: false }),
        area,
    );
}
