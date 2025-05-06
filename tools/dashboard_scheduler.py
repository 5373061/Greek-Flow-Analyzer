"""
Dashboard Scheduler - Extension module for the IntegratedDashboard class
Adds scheduling capabilities to the trade dashboard.

Usage:
    from tools.dashboard_scheduler import add_scheduling_to_dashboard
    dashboard = IntegratedDashboard(root)
    add_scheduling_to_dashboard(dashboard)
"""

import logging
from datetime import datetime, time, timedelta

logger = logging.getLogger("DashboardScheduler")

def add_scheduling_to_dashboard(dashboard):
    """Add scheduling methods to an existing dashboard instance."""
    logger = logging.getLogger("DashboardScheduler")
    
    # Add the schedule_refresh method
    def schedule_refresh(seconds_delay):
        """Schedule a data refresh after a specified delay in seconds."""
        if hasattr(dashboard, 'root') and dashboard.root:
            # Check if dashboard has refresh_data method
            if not hasattr(dashboard, 'refresh_data'):
                logger.warning("Dashboard does not have refresh_data method. Adding a basic implementation.")
                # Add a basic refresh_data method
                def basic_refresh():
                    logger.info("Basic refresh triggered")
                    if hasattr(dashboard, 'load_all_data'):
                        dashboard.load_all_data()
                        logger.info("Reloaded all data")
                    if hasattr(dashboard, 'update_recommendation_list'):
                        dashboard.update_recommendation_list()
                        logger.info("Updated recommendation list")
                    if hasattr(dashboard, 'update_status'):
                        dashboard.update_status("Data refreshed", "info")
                
                dashboard.refresh_data = basic_refresh
            
            # Schedule the refresh
            dashboard.root.after(seconds_delay * 1000, dashboard.refresh_data)
            logger.info(f"Scheduled data refresh in {seconds_delay} seconds.")
            
            # Update status with scheduled time
            refresh_time = datetime.now().timestamp() + seconds_delay
            refresh_time_str = datetime.fromtimestamp(refresh_time).strftime('%I:%M %p')
            
            if hasattr(dashboard, 'update_status'):
                dashboard.update_status(f"Next refresh scheduled at {refresh_time_str}", "info")
            else:
                logger.info(f"Next refresh scheduled at {refresh_time_str}")
    
    # Add the method to the dashboard instance
    dashboard.schedule_refresh = schedule_refresh
    
    # We don't need to add a refresh button since it already exists
    dashboard.add_refresh_button = lambda: logger.info("Manual refresh button is already available.")
    
    # Add ability to set up scheduled refreshes at specific times
    def setup_daily_refreshes(refresh_times=None):
        """Set up refreshes at specific times each day."""
        if refresh_times is None:
            # Default times: 9:30 AM, 12:30 PM, 4:00 PM
            refresh_times = [
                time(9, 30),   # 9:30 AM
                time(12, 30),  # 12:30 PM
                time(16, 0)    # 4:00 PM
            ]
        
        now = datetime.now()
        
        # Schedule all refreshes for today or tomorrow if the time has passed
        for refresh_time in refresh_times:
            # Combine date and time
            refresh_datetime = datetime.combine(now.date(), refresh_time)
            
            # If time already passed today, schedule for tomorrow
            if refresh_datetime < now:
                # Fix: Use datetime.timedelta instead of datetime.datetime.timedelta
                from datetime import timedelta
                tomorrow = now.date() + timedelta(days=1)
                refresh_datetime = datetime.combine(tomorrow, refresh_time)
            
            # Calculate seconds until refresh
            seconds_until_refresh = (refresh_datetime - now).total_seconds()
            
            # Schedule the refresh
            schedule_refresh(int(seconds_until_refresh))
    
    # Add the method to the dashboard instance
    dashboard.setup_daily_refreshes = setup_daily_refreshes
    
    return dashboard

# Define refresh times (24-hour format)
from datetime import time
refresh_times = {
    "morning": [time(9, 30)],                # 9:30 AM
    "midday": [time(12, 30)],                # 12:30 PM
    "evening": [time(16, 0)],                # 4:00 PM
    "all": [time(9, 30), time(12, 30), time(16, 0)]  # All times
}

if 'dashboard_instance' in locals() and hasattr(dashboard_instance, 'refresh_schedule') and dashboard_instance.refresh_schedule in refresh_times:
    dashboard_instance.setup_daily_refreshes(refresh_times[dashboard_instance.refresh_schedule])
    logger.info(f"Set up {dashboard_instance.refresh_schedule} refresh schedule")








